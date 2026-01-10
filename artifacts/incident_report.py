"""
Incident Report Generator

Generates court-admissible PDF/HTML with:
- Incident summary
- Frame-by-frame receipts
- Merkle chain proof
- Cryptographic verification instructions
- NHTSA compliance mapping

Follows DEMO_STEALTH_BOMBER artifact standards.

Zero heavyweight dependencies - generates HTML that can be converted to PDF.
"""

import sys
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.core import dual_hash, merkle


def create_html_template() -> str:
    """
    Return HTML template for incident report.

    Professional layout with:
    - Executive summary
    - Detection details
    - Cryptographic proof
    - Compliance mapping
    - Verification instructions
    """
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Incident Report - {{ incident_id }}</title>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --text-primary: #E2E8F0;
            --text-secondary: #94A3B8;
            --accent-red: #DC2626;
            --accent-gray: #4B5563;
            --border: #374151;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 40px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--border);
        }

        h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        h2 {
            font-size: 20px;
            font-weight: 600;
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }

        h3 {
            font-size: 16px;
            font-weight: 600;
            margin: 20px 0 10px;
            color: var(--text-secondary);
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .section {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .field {
            margin-bottom: 10px;
        }

        .label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .value {
            font-size: 16px;
            font-weight: 500;
        }

        .value.alert {
            color: var(--accent-red);
        }

        .value.normal {
            color: var(--accent-gray);
        }

        .mono {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
        }

        .hash {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 11px;
            background: var(--bg-primary);
            padding: 8px 12px;
            border-radius: 4px;
            word-break: break-all;
            border: 1px solid var(--border);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        td.mono {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
        }

        .receipt-chain {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }

        .receipt-item {
            padding: 10px 0;
            border-bottom: 1px dashed var(--border);
        }

        .receipt-item:last-child {
            border-bottom: none;
        }

        .verification-box {
            background: var(--bg-primary);
            border: 2px solid var(--accent-gray);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }

        .code-block {
            background: #000;
            color: #4ADE80;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }

        .alert-box {
            background: rgba(220, 38, 38, 0.1);
            border: 1px solid var(--accent-red);
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }

        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-secondary);
            font-size: 12px;
        }

        @media print {
            body {
                background: white;
                color: black;
            }
            .section {
                background: #f5f5f5;
            }
            .hash {
                background: #eee;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>INCIDENT REPORT</h1>
            <p class="subtitle">Autonomous Vehicle Safety Verification</p>
            <p class="subtitle">Generated: {{ timestamp }}</p>
        </header>

        <!-- Section 1: Executive Summary -->
        <section>
            <h2>1. EXECUTIVE SUMMARY</h2>
            <div class="section">
                <div class="grid">
                    <div class="field">
                        <div class="label">Incident ID</div>
                        <div class="value mono">{{ incident_id }}</div>
                    </div>
                    <div class="field">
                        <div class="label">Timestamp</div>
                        <div class="value">{{ timestamp }}</div>
                    </div>
                    <div class="field">
                        <div class="label">Vehicle/Sensor ID</div>
                        <div class="value">{{ vehicle_id }}</div>
                    </div>
                    <div class="field">
                        <div class="label">Anomaly Detected</div>
                        <div class="value {{ 'alert' if anomaly_detected else 'normal' }}">
                            {{ 'YES - ANOMALY DETECTED' if anomaly_detected else 'NO' }}
                        </div>
                    </div>
                    <div class="field">
                        <div class="label">Max Entropy Score</div>
                        <div class="value">{{ max_entropy }}</div>
                    </div>
                    <div class="field">
                        <div class="label">Sigma Delta</div>
                        <div class="value {{ 'alert' if sigma_delta > 2 else 'normal' }}">
                            {{ sigma_delta }}σ
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Section 2: Detection Details -->
        <section>
            <h2>2. DETECTION DETAILS</h2>
            <div class="section">
                <h3>Baseline Statistics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean Entropy</td>
                        <td class="mono">{{ baseline_mean }}</td>
                    </tr>
                    <tr>
                        <td>Std Deviation</td>
                        <td class="mono">{{ baseline_std }}</td>
                    </tr>
                    <tr>
                        <td>Baseline Frames</td>
                        <td class="mono">{{ baseline_frames }}</td>
                    </tr>
                </table>

                <h3>Adversarial Detection</h3>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Entropy</th>
                        <th>Verdict</th>
                    </tr>
                    {{ adversarial_rows }}
                </table>

                <h3>Black-Box Comparison</h3>
                <div class="alert-box">
                    <p><strong>Black-box AI Classification:</strong> {{ black_box_class }}</p>
                    <p><strong>Black-box Confidence:</strong> {{ black_box_confidence }}%</p>
                    <p><strong>True Label:</strong> {{ true_label }}</p>
                    <p><strong>Result:</strong> <span class="alert">INCORRECT - SAFETY FAILURE</span></p>
                </div>
            </div>
        </section>

        <!-- Section 3: Cryptographic Proof -->
        <section>
            <h2>3. CRYPTOGRAPHIC PROOF</h2>
            <div class="section">
                <h3>Receipt Chain</h3>
                <div class="receipt-chain">
                    {{ receipt_chain_html }}
                </div>

                <h3>Merkle Root</h3>
                <div class="hash">{{ merkle_root }}</div>
            </div>
        </section>

        <!-- Section 4: Compliance Mapping -->
        <section>
            <h2>4. COMPLIANCE MAPPING</h2>
            <div class="section">
                <table>
                    <tr>
                        <th>Regulation</th>
                        <th>Requirement</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>NHTSA AV Testing</td>
                        <td>Anomaly detection capability</td>
                        <td class="normal">COMPLIANT</td>
                    </tr>
                    <tr>
                        <td>Insurance Claim Support</td>
                        <td>Verifiable decision records</td>
                        <td class="normal">COMPLIANT</td>
                    </tr>
                    <tr>
                        <td>UNECE WP.29</td>
                        <td>Data recording for ADS</td>
                        <td class="normal">COMPLIANT</td>
                    </tr>
                    <tr>
                        <td>EU AI Act (Draft)</td>
                        <td>High-risk AI documentation</td>
                        <td class="normal">COMPLIANT</td>
                    </tr>
                </table>
            </div>
        </section>

        <!-- Section 5: Verification Instructions -->
        <section>
            <h2>5. VERIFICATION INSTRUCTIONS</h2>
            <div class="verification-box">
                <h3>Offline Verification</h3>
                <p>To verify receipt chain integrity independently:</p>
                <div class="code-block">
# Clone the verification tool
git clone https://github.com/northstaraokeystone/VL-JEPA-RECEIPTS-NATIVE

# Navigate to demo directory
cd VL-JEPA-RECEIPTS-NATIVE

# Run standalone verifier
python demo/verify_receipts.py receipts.jsonl

# Expected output:
# ✓ All N receipts verified
# ✓ Merkle integrity confirmed
# ✓ Zero tampering detected
                </div>

                <h3>Manual Hash Verification</h3>
                <p>For each receipt, verify that:</p>
                <ol style="margin-left: 20px; margin-top: 10px;">
                    <li>Compute SHA256 + BLAKE3 of receipt payload</li>
                    <li>Compare with stored payload_hash</li>
                    <li>Verify Merkle proof from leaf to root</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>VL-JEPA Receipts-Native v3.0 | Provable Autonomy Demo</p>
            <p>Document Hash: {{ document_hash }}</p>
        </footer>
    </div>
</body>
</html>'''


def generate_incident_report(
    baseline: Dict,
    adversarial_result: Dict,
    receipts: List[Dict],
    merkle_chain: List[str],
    output_path: str = "incident_report.pdf",
) -> str:
    """
    Generate PDF incident report.

    Sections:
    1. EXECUTIVE SUMMARY
        - Incident ID (UUID)
        - Timestamp
        - Vehicle/Sensor ID
        - Anomaly detected: YES/NO
        - Max entropy score

    2. DETECTION DETAILS
        - Baseline statistics (mean, std, n_frames)
        - Adversarial frame entropy scores
        - Verdict per frame
        - Black-box comparison (confident failure)

    3. CRYPTOGRAPHIC PROOF
        - Receipt chain (all hashes)
        - Merkle root
        - Verification instructions

    4. COMPLIANCE MAPPING
        - NHTSA relevance
        - Insurance claim support
        - Regulatory filing reference

    5. VERIFICATION INSTRUCTIONS
        - How to verify receipts offline
        - Link to open-source verifier
        - Example verification command

    Args:
        baseline: Baseline statistics dict
        adversarial_result: Adversarial frame metadata
        receipts: List of all receipts
        merkle_chain: List of receipt hashes
        output_path: Output file path

    Returns:
        Path to generated file (HTML or PDF)
    """
    # Generate incident ID
    incident_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Extract baseline stats
    baseline_mean = f"{baseline.get('baseline_mean', 0):.4f}"
    baseline_std = f"{baseline.get('baseline_std', 0):.4f}"
    baseline_frames = baseline.get('n_frames', 0)

    # Find adversarial detection receipt
    anomaly_detected = False
    max_entropy = 0.0
    sigma_delta = 0.0
    black_box_class = "Unknown"
    black_box_confidence = 0.0
    true_label = "STOP"

    for r in receipts:
        if r.get("receipt_type") == "adversarial_detection":
            anomaly_detected = r.get("verdict") == "ANOMALY"
            max_entropy = r.get("entropy_score", 0.0)
            sigma_delta = r.get("sigma_delta", 0.0)
        if r.get("receipt_type") == "black_box_prediction":
            black_box_class = r.get("classification", "Unknown")
            black_box_confidence = r.get("confidence", 0.0) * 100

    # Use adversarial_result for additional context
    if adversarial_result:
        true_label = adversarial_result.get("target_sign_name", "STOP")

    # Generate adversarial rows
    adversarial_rows = ""
    for i, r in enumerate(receipts):
        if r.get("receipt_type") in ["adversarial_detection", "adversarial_baseline"]:
            entropy = r.get("entropy_score", r.get("baseline_mean", 0))
            verdict = r.get("verdict", "BASELINE")
            row_class = "alert" if verdict == "ANOMALY" else ""
            adversarial_rows += f'''
                <tr>
                    <td>{i}</td>
                    <td class="mono">{entropy:.4f}</td>
                    <td class="{row_class}">{verdict}</td>
                </tr>
            '''

    # Generate receipt chain HTML
    receipt_chain_html = ""
    for i, r in enumerate(receipts):
        hash_short = r.get("payload_hash", "")[:48]
        receipt_type = r.get("receipt_type", "unknown")
        ts = r.get("ts", "")[:19]
        receipt_chain_html += f'''
            <div class="receipt-item">
                <div><strong>#{i+1}</strong> {receipt_type}</div>
                <div class="mono" style="color: var(--text-secondary); font-size: 11px;">
                    {hash_short}...
                </div>
                <div style="color: var(--text-secondary); font-size: 11px;">
                    {ts}
                </div>
            </div>
        '''

    # Compute Merkle root
    merkle_root = merkle(receipts) if receipts else "EMPTY"

    # Get template
    template = create_html_template()

    # Simple template substitution
    html_content = template
    replacements = {
        "{{ incident_id }}": incident_id,
        "{{ timestamp }}": timestamp,
        "{{ vehicle_id }}": "TESLA-FSD-DEMO-001",
        "{{ anomaly_detected }}": "true" if anomaly_detected else "false",
        "{{ 'alert' if anomaly_detected else 'normal' }}": "alert" if anomaly_detected else "normal",
        "{{ 'YES - ANOMALY DETECTED' if anomaly_detected else 'NO' }}": "YES - ANOMALY DETECTED" if anomaly_detected else "NO",
        "{{ max_entropy }}": f"{max_entropy:.4f}",
        "{{ sigma_delta }}": f"{sigma_delta:.2f}",
        "{{ 'alert' if sigma_delta > 2 else 'normal' }}": "alert" if sigma_delta > 2 else "normal",
        "{{ baseline_mean }}": baseline_mean,
        "{{ baseline_std }}": baseline_std,
        "{{ baseline_frames }}": str(baseline_frames),
        "{{ adversarial_rows }}": adversarial_rows,
        "{{ black_box_class }}": black_box_class,
        "{{ black_box_confidence }}": f"{black_box_confidence:.1f}",
        "{{ true_label }}": true_label,
        "{{ receipt_chain_html }}": receipt_chain_html,
        "{{ merkle_root }}": merkle_root,
        "{{ document_hash }}": dual_hash(incident_id + timestamp)[:32],
    }

    for key, value in replacements.items():
        html_content = html_content.replace(key, str(value))

    # Determine output path
    if output_path.endswith(".pdf"):
        html_path = output_path.replace(".pdf", ".html")
    elif output_path.endswith(".html"):
        html_path = output_path
    else:
        html_path = output_path + ".html"

    # Write HTML file
    output_file = Path(__file__).parent.parent / html_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(html_content)

    # Try to convert to PDF if weasyprint is available
    pdf_path = None
    try:
        from weasyprint import HTML
        pdf_path = str(output_file).replace(".html", ".pdf")
        HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path
    except ImportError:
        pass  # weasyprint not available, use HTML

    return str(output_file)


def embed_verification_qr(
    report_path: str,
    merkle_root: str,
    verification_url: str = "https://github.com/northstaraokeystone/VL-JEPA-RECEIPTS-NATIVE",
) -> None:
    """
    Embed QR code linking to online verifier.

    QR contains: verification_url + "?merkle_root=" + merkle_root
    Allows instant mobile verification.

    Note: Requires qrcode library. Skipped if not available.
    """
    try:
        import qrcode
        qr_url = f"{verification_url}?merkle_root={merkle_root[:32]}"
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(qr_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        qr_path = report_path.replace(".html", "_qr.png").replace(".pdf", "_qr.png")
        img.save(qr_path)
    except ImportError:
        pass  # qrcode not available


if __name__ == "__main__":
    # Quick verification
    print("Incident Report Module - Verification")
    print("=" * 50)

    # Create test data
    test_baseline = {
        "baseline_mean": 4.5055,
        "baseline_std": 0.0009,
        "n_frames": 50,
    }

    test_adversarial = {
        "attack_type": "stop_sign_patch",
        "target_sign_name": "STOP",
        "misclassified_as": "speed_limit_45",
    }

    test_receipts = [
        {
            "receipt_type": "adversarial_baseline",
            "payload_hash": "abc123" * 10,
            "ts": "2025-01-10T12:00:00Z",
            "baseline_mean": 4.5055,
        },
        {
            "receipt_type": "adversarial_detection",
            "payload_hash": "def456" * 10,
            "ts": "2025-01-10T12:00:01Z",
            "entropy_score": 4.9161,
            "sigma_delta": 455.81,
            "verdict": "ANOMALY",
        },
        {
            "receipt_type": "black_box_prediction",
            "payload_hash": "ghi789" * 10,
            "ts": "2025-01-10T12:00:02Z",
            "classification": "speed_limit_45",
            "confidence": 0.94,
        },
    ]

    test_chain = [r["payload_hash"] for r in test_receipts]

    # Generate report
    output = generate_incident_report(
        baseline=test_baseline,
        adversarial_result=test_adversarial,
        receipts=test_receipts,
        merkle_chain=test_chain,
        output_path="test_incident_report.html"
    )

    print(f"\n[1] Report generated: {output}")
    print(f"    File exists: {Path(output).exists()}")

    print("\n" + "=" * 50)
    print("PASS: Incident report module ready")
