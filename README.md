# Zero-Shot Enterprise Attack Detection

A local Streamlit dashboard for evaluating novel-attack **anomaly detection** on
enterprise user behavior logs. The model learns only from known-normal rows in
`train_seen`; `test_unseen` attack labels are held out and used only to evaluate
the detector. User identifiers, timestamps, labels, and split names are never
model features.

## Dataset format

The CSV must include:

- Features: `login_hour`, `failed_logins_24h`, `files_accessed_24h`,
  `usb_connected`, `external_email_count_24h`, `bytes_uploaded_mb_24h`,
  `privilege_change`, `geo_anomaly`, `unique_hosts_accessed_24h`,
  `sensitive_file_access_24h`, `department`, `event_type`
- Evaluation fields: `attack_label`, `zero_shot_split`
- `zero_shot_split` must include `train_seen` and `test_unseen`.
- Attack labels in `test_unseen` must not also occur in `train_seen`.

The model uses a reproducible 60/20/20 split of `train_seen` normal rows:
training, threshold calibration, and held-out normal evaluation. It then
compares held-out normals with unseen test attacks. The false-positive target
sets the score threshold using only the calibration normals; actual results can
differ.

This is novelty detection, not semantic zero-shot classification. It does not
infer attack names or use external attack descriptions. Metrics depend on the
provided dataset and this evaluation split.

## Run locally

1. Open a terminal in this project folder.
2. Create and activate an isolated environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

   By default, the app looks for
   `~/Downloads/zero_shot_enterprise_user_behaviour_logs (1).csv`.
   To select another local file, set `ZERO_SHOT_DATASET` before launching:

   ```powershell
   $env:ZERO_SHOT_DATASET = "C:\path\to\your\logs.csv"
   streamlit run app.py
   ```

   Or use the CSV uploader in the dashboard: select a compatible CSV and click
   **Deploy dataset** to run detection and refresh the results table. Use
   **Download active dataset** to save the currently loaded CSV. These actions
   work in the running local app; they do not publish the app or upload the
   dataset to a hosting service. The dashboard prints its local URL (usually
   `http://localhost:8501`) in the terminal.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Privacy and deployment

The raw dataset is not copied into this project or uploaded by the app. Uploaded
data is kept in the running app's session to produce its results. A local
Streamlit URL is only accessible from this machine. A public deployment needs a
hosting account and a deliberate data-hosting decision; do not commit enterprise
logs or credentials to a public repository or deployment.
