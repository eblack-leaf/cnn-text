// ── Dataset loaders ───────────────────────────────────────────────────────────
//
// Each loader normalises its native format into plain `(label, text)` pairs.
// `DatasetKind::load` returns `(train_pairs, test_pairs_opt)`.
// When a test split is pre-provided (Amazon) it is used as the validation set;
// otherwise `val_ratio` in TrainingConfig controls the split.

#[derive(Clone, Debug, Default)]
pub enum DatasetKind {
    /// Headerless `label,text` CSV — the project's existing format.
    #[default]
    Custom,
    /// amazon_review_polarity_csv/ directory with train.csv / test.csv.
    /// Labels: 1 → "negative", 2 → "positive". Columns: polarity,title,review.
    Amazon,
    /// SMSSpamCollection file — tab-separated `label<TAB>text`.
    /// Labels: "ham" / "spam".
    Sms,
    /// "IMDB Dataset.csv" with header `review,sentiment`.
    /// Labels: "positive" / "negative". Reviews may contain HTML `<br />`.
    Imdb,
}

impl DatasetKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "custom" => Some(Self::Custom),
            "amazon" => Some(Self::Amazon),
            "sms"    => Some(Self::Sms),
            "imdb"   => Some(Self::Imdb),
            _        => None,
        }
    }

    /// Default path/directory for this dataset relative to the project root.
    pub fn default_path(&self) -> &'static str {
        match self {
            Self::Custom => "data/dataset.csv",
            Self::Amazon => "data/amazon_review_polarity_csv",
            Self::Sms    => "data/sms+spam+collection/SMSSpamCollection",
            Self::Imdb   => "data/archive/IMDB Dataset.csv",
        }
    }

    /// Load `(train_pairs, test_pairs_opt)`.
    /// `path` is a file path for Custom/Sms/Imdb, a directory for Amazon.
    pub fn load(&self, path: &str) -> (Vec<(String, String)>, Option<Vec<(String, String)>>) {
        match self {
            Self::Custom => (load_custom_csv(path), None),
            Self::Amazon => load_amazon(path),
            Self::Sms    => (load_sms(path), None),
            Self::Imdb   => (load_imdb(path), None),
        }
    }
}

// ── Custom (existing format) ───────────────────────────────────────────────────

/// Headerless `label,text` CSV — the project's native format.
fn load_custom_csv(path: &str) -> Vec<(String, String)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    content
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ',');
            let label = parts.next()?.trim().to_string();
            let text  = parts.next()?.trim().to_string();
            if label.is_empty() || text.is_empty() { return None; }
            Some((label, text))
        })
        .collect()
}

// ── Amazon Review Polarity ────────────────────────────────────────────────────

fn load_amazon(dir: &str) -> (Vec<(String, String)>, Option<Vec<(String, String)>>) {
    let train_path = format!("{dir}/train.csv");
    let test_path  = format!("{dir}/test.csv");
    println!("Loading Amazon train from {train_path} …");
    let train = load_amazon_file(&train_path);
    println!("Loading Amazon test  from {test_path} …");
    let test  = load_amazon_file(&test_path);
    println!("Amazon: {} train / {} test pairs", train.len(), test.len());
    (train, Some(test))
}

fn load_amazon_file(path: &str) -> Vec<(String, String)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    content
        .lines()
        .filter_map(|line| {
            let fields = parse_quoted_csv_row(line);
            if fields.len() < 3 { return None; }
            let label = match fields[0].trim() {
                "1" => "negative",
                "2" => "positive",
                _   => return None,
            };
            // Concatenate title + review body.
            let text = format!("{} {}", fields[1].trim(), fields[2].trim());
            Some((label.to_string(), text))
        })
        .collect()
}

// ── SMS Spam Collection ───────────────────────────────────────────────────────

fn load_sms(path: &str) -> Vec<(String, String)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    content
        .lines()
        .filter_map(|line| {
            let (label, text) = line.split_once('\t')?;
            let label = label.trim();
            let text  = text.trim();
            if label.is_empty() || text.is_empty() { return None; }
            Some((label.to_string(), text.to_string()))
        })
        .collect()
}

// ── IMDB ─────────────────────────────────────────────────────────────────────

fn load_imdb(path: &str) -> Vec<(String, String)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    let mut lines = content.lines();
    lines.next(); // skip header row
    lines
        .filter_map(|line| {
            // Format: "review text",sentiment  (review is double-quoted)
            let fields = parse_quoted_csv_row(line);
            if fields.len() < 2 { return None; }
            let review    = strip_html(fields[0].trim());
            let sentiment = fields[1].trim().to_string();
            if review.is_empty() || sentiment.is_empty() { return None; }
            Some((sentiment, review))
        })
        .collect()
}

// ── CSV helpers ───────────────────────────────────────────────────────────────

/// Parse one RFC-4180 CSV row: handles double-quoted fields and `""` escapes.
fn parse_quoted_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut chars  = line.chars().peekable();

    loop {
        match chars.peek() {
            None => break,
            Some(&',') => { chars.next(); } // consume separator before next field
            _ => {}
        }
        if chars.peek().is_none() { break; }

        if chars.peek() == Some(&'"') {
            chars.next(); // consume opening quote
            let mut field = String::new();
            loop {
                match chars.next() {
                    None      => break,
                    Some('"') => {
                        if chars.peek() == Some(&'"') {
                            chars.next(); // escaped "" → one literal "
                            field.push('"');
                        } else {
                            break; // closing quote
                        }
                    }
                    Some(c) => field.push(c),
                }
            }
            fields.push(field);
        } else {
            // Unquoted field: read until next comma.
            let field: String = chars.by_ref().take_while(|&c| c != ',').collect();
            fields.push(field);
        }
    }
    fields
}

/// Remove HTML tags and normalise whitespace.
fn strip_html(s: &str) -> String {
    let mut out    = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => { in_tag = false; out.push(' '); }
            _   => if !in_tag { out.push(c); }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
