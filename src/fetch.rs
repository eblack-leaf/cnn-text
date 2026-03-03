// ── AG News ───────────────────────────────────────────────────────────────────

const AGNEWS_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
];
const AGNEWS_LABELS: &[&str] = &["", "World", "Sports", "Business", "SciTech"];

pub fn agnews() {
    use std::io::{BufRead, Write};

    std::fs::create_dir_all("data").expect("cannot create data/");
    let mut out = std::fs::File::create("data/dataset.csv")
        .expect("cannot create dataset.csv");

    let mut total = 0usize;
    for &url in AGNEWS_URLS {
        eprintln!("Downloading {url} …");
        let response = ureq::get(url).call().expect("HTTP request failed");
        let reader = std::io::BufReader::new(response.into_body().into_reader());

        for line in reader.lines() {
            let line = line.expect("IO error reading response");
            let fields = parse_quoted_csv(&line);
            if fields.len() < 3 { continue; }

            let idx: usize = fields[0].trim().parse().expect("class index not a number");
            let label = AGNEWS_LABELS[idx];
            let text  = format!("{}. {}", fields[1].trim(), fields[2].trim())
                .replace('\n', " ")
                .replace('\r', " ");

            writeln!(out, "{label},{text}").unwrap();
            total += 1;
        }
        eprintln!("  {} rows so far", total);
    }

    eprintln!("Written {total} rows → data/dataset.csv");
}

/// Parse one line of the AG News CSV where every field is double-quoted.
fn parse_quoted_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut chars  = line.chars().peekable();

    while chars.peek().is_some() {
        if chars.peek() == Some(&'"') {
            chars.next();
            let mut field = String::new();
            loop {
                match chars.next() {
                    None => break,
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
            if chars.peek() == Some(&',') { chars.next(); }
        } else {
            let field: String = chars.by_ref().take_while(|&c| c != ',').collect();
            fields.push(field);
        }
    }

    fields
}

// ── SMS Spam Collection (UCI ML Repository) ───────────────────────────────────

const SMS_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip";
const SMS_OUT: &str = "data/sms+spam+collection/SMSSpamCollection";

pub fn sms() {
    use std::io::Read;

    if std::path::Path::new(SMS_OUT).exists() {
        eprintln!("{SMS_OUT} already exists — skipping download.");
        return;
    }

    std::fs::create_dir_all("data/sms+spam+collection").expect("cannot create output dir");

    eprintln!("Downloading SMS Spam Collection from UCI …");
    let response = ureq::get(SMS_URL).call().expect("HTTP request failed");
    let mut data = Vec::new();
    response.into_body().into_reader().read_to_end(&mut data).expect("read error");
    eprintln!("  {:.1} KB", data.len() as f64 / 1_000.0);

    eprintln!("Extracting SMSSpamCollection …");
    let cursor  = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).expect("invalid zip");

    // Locate the data file — it may be at the root or inside a subdirectory.
    let entry_name = (0..archive.len())
        .find_map(|i| {
            let e = archive.by_index(i).ok()?;
            let name = e.name().to_string();
            if name.contains("SMSSpamCollection") && !name.ends_with('/') {
                Some(name)
            } else {
                None
            }
        })
        .expect("SMSSpamCollection not found in zip");

    let mut entry = archive.by_name(&entry_name).unwrap();
    let mut out   = std::fs::File::create(SMS_OUT).expect("cannot create output file");
    std::io::copy(&mut entry, &mut out).expect("extraction failed");

    eprintln!("Saved → {SMS_OUT}");
}

// ── IMDB (Stanford Large Movie Review Dataset) ────────────────────────────────
//
// Source: http://ai.stanford.edu/~amaas/data/sentiment/
// The tar.gz contains aclImdb/{train,test}/{pos,neg}/*.txt — one review per file.
// We merge everything into data/archive/IMDB Dataset.csv (header: review,sentiment)
// so the existing `load_imdb` loader picks it up without changes.

const IMDB_URL: &str =
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
const IMDB_OUT: &str = "data/archive/IMDB Dataset.csv";

pub fn imdb() {
    use std::io::{Read, Write};

    if std::path::Path::new(IMDB_OUT).exists() {
        eprintln!("{IMDB_OUT} already exists — skipping download.");
        return;
    }

    std::fs::create_dir_all("data/archive").expect("cannot create output dir");

    eprintln!("Downloading Stanford IMDB dataset (~84 MB) …");
    let response = ureq::get(IMDB_URL).call().expect("HTTP request failed");
    let mut reader = std::io::BufReader::new(response.into_body().into_reader());

    // Stream directly through gzip → tar without writing to disk.
    let gz      = flate2::read::GzDecoder::new(&mut reader);
    let mut tar = tar::Archive::new(gz);

    let mut out = std::fs::File::create(IMDB_OUT).expect("cannot create output file");
    writeln!(out, "review,sentiment").unwrap();

    let mut total = 0usize;

    for entry in tar.entries().expect("cannot read tar") {
        let mut entry = entry.expect("corrupt tar entry");
        let path = entry.path().expect("bad path").to_path_buf();

        // We only want: aclImdb/{train,test}/{pos,neg}/*.txt
        let parts: Vec<&str> = path
            .components()
            .map(|c| c.as_os_str().to_str().unwrap_or(""))
            .collect();

        if parts.len() < 4 { continue; }
        if parts[0] != "aclImdb" { continue; }
        // parts[1]: "train" | "test"
        // parts[2]: "pos" | "neg"  (skip "unsup", "urls", etc.)
        let sentiment = match parts[2] {
            "pos" => "positive",
            "neg" => "negative",
            _     => continue,
        };
        if !parts[3].ends_with(".txt") { continue; }

        let mut content = String::new();
        entry.read_to_string(&mut content).expect("read error");

        // Escape double-quotes and strip newlines for CSV.
        let escaped = content
            .replace('"', "\"\"")
            .replace(['\n', '\r'], " ");

        writeln!(out, "\"{escaped}\",{sentiment}").unwrap();
        total += 1;

        if total % 5_000 == 0 {
            eprint!("\r  {total} reviews written…");
        }
    }

    eprintln!("\r  {total} reviews written.");
    eprintln!("Saved → {IMDB_OUT}");
}

// ── GloVe ─────────────────────────────────────────────────────────────────────

const GLOVE_URL:      &str = "https://nlp.stanford.edu/data/glove.6B.zip";
const GLOVE_ZIP_PATH: &str = "data/glove.6B.zip";

pub fn glove(dim: u32) {
    use std::io::{Read, Write};

    assert!(
        [50, 100, 200, 300].contains(&dim),
        "dimension must be one of: 50 100 200 300"
    );

    let entry_name = format!("glove.6B.{dim}d.txt");
    let out_path   = format!("data/{entry_name}");

    if std::path::Path::new(&out_path).exists() {
        eprintln!("{out_path} already exists — skipping download.");
        return;
    }

    std::fs::create_dir_all("data").expect("cannot create data/");

    eprintln!("Downloading {GLOVE_URL} (~822 MB) …");
    let response = ureq::get(GLOVE_URL).call().expect("HTTP request failed");
    let mut reader = std::io::BufReader::new(response.into_body().into_reader());
    let mut zip_file = std::fs::File::create(GLOVE_ZIP_PATH).expect("cannot create zip file");

    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    loop {
        let n = reader.read(&mut buf).expect("read error");
        if n == 0 { break; }
        zip_file.write_all(&buf[..n]).expect("write error");
        downloaded += n as u64;
        eprint!("\r  {:.1} MB", downloaded as f64 / 1_000_000.0);
    }
    eprintln!("\r  {:.1} MB — done", downloaded as f64 / 1_000_000.0);
    drop(zip_file);

    eprintln!("Extracting {entry_name} …");
    let zip_file = std::fs::File::open(GLOVE_ZIP_PATH).expect("cannot open zip");
    let mut archive = zip::ZipArchive::new(zip_file).expect("invalid zip");
    let mut entry = archive.by_name(&entry_name)
        .unwrap_or_else(|_| panic!("{entry_name} not found in zip"));
    let mut out = std::fs::File::create(&out_path).expect("cannot create output file");
    std::io::copy(&mut entry, &mut out).expect("extraction failed");
    eprintln!("Saved → {out_path}");

    std::fs::remove_file(GLOVE_ZIP_PATH).expect("could not delete zip");
    eprintln!("Deleted {GLOVE_ZIP_PATH}");
}
