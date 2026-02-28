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
