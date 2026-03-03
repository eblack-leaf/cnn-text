// ── Dataset loaders ───────────────────────────────────────────────────────────
//
// Each loader normalises its native format into plain `(label, text)` pairs.
// `DatasetKind::load` returns `(train_pairs, test_pairs_opt)`.
// test_pairs_opt is always None here — val split is controlled by `val_ratio`.

#[derive(Clone, Debug, Default)]
pub enum DatasetKind {
    /// Headerless `label,text` CSV — the project's native format.
    /// AG News (fetched via `fetch-agnews`) writes to this format.
    #[default]
    AgNews,
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
            "agnews" | "custom" => Some(Self::AgNews),
            "sms"               => Some(Self::Sms),
            "imdb"              => Some(Self::Imdb),
            _                   => None,
        }
    }

    /// Default path for this dataset relative to the project root.
    pub fn default_path(&self) -> &'static str {
        match self {
            Self::AgNews => "data/dataset.csv",
            Self::Sms    => "data/sms+spam+collection/SMSSpamCollection",
            Self::Imdb   => "data/archive/IMDB Dataset.csv",
        }
    }

    /// Load `(train_pairs, test_pairs_opt)`.
    pub fn load(&self, path: &str) -> (Vec<(String, String)>, Option<Vec<(String, String)>>) {
        match self {
            Self::AgNews => (load_csv(path), None),
            Self::Sms    => (load_sms(path), None),
            Self::Imdb   => (load_imdb(path), None),
        }
    }
}

// ── AG News / generic headerless CSV ─────────────────────────────────────────

/// Headerless `label,text` CSV — the project's native format.
fn load_csv(path: &str) -> Vec<(String, String)> {
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
            Some(&',') => { chars.next(); }
            _ => {}
        }
        if chars.peek().is_none() { break; }

        if chars.peek() == Some(&'"') {
            chars.next();
            let mut field = String::new();
            loop {
                match chars.next() {
                    None      => break,
                    Some('"') => {
                        if chars.peek() == Some(&'"') {
                            chars.next();
                            field.push('"');
                        } else {
                            break;
                        }
                    }
                    Some(c) => field.push(c),
                }
            }
            fields.push(field);
        } else {
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
