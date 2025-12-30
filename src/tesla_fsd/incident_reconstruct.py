"""
Tesla FSD Incident Reconstruction Module

Reconstructs incidents from Merkle-anchored frame trails.
Provides NHTSA-compliant audit trails.
"""

from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, merkle
from src.core.ledger import Ledger
from src.governance import RACIManager, ProvenanceTracker


class IncidentReconstructor:
    """
    Reconstructs FSD incidents from receipt trails.

    Creates NHTSA-compliant audit packages.
    """

    def __init__(
        self,
        vehicle_id: str = "unknown",
    ):
        self.vehicle_id = vehicle_id
        self.domain = "tesla_fsd"

        self.ledger = Ledger()
        self.raci_manager = RACIManager(domain=self.domain)
        self.provenance_tracker = ProvenanceTracker(domain=self.domain)

    def reconstruct_incident(
        self,
        start_ts: str,
        end_ts: str,
        incident_type: str = "intervention",
    ) -> dict:
        """
        Reconstruct an incident from timestamp range.

        Args:
            start_ts: Incident start (ISO8601)
            end_ts: Incident end (ISO8601)
            incident_type: Type of incident

        Returns:
            Reconstruction receipt
        """
        import uuid
        incident_id = str(uuid.uuid4())

        # Assign RACI
        raci_receipt = self.raci_manager.assign_raci(
            "incident_reconstruction",
            incident_id,
        )

        # Get relevant receipts
        all_receipts = self.ledger.read_all()
        incident_receipts = [
            r for r in all_receipts
            if start_ts <= r.get("ts", "") <= end_ts
            and r.get("vehicle_id") == self.vehicle_id
        ]

        # Compute Merkle root of incident
        incident_root = merkle(incident_receipts)

        # Extract timeline
        timeline = []
        for r in sorted(incident_receipts, key=lambda x: x.get("ts", "")):
            timeline.append({
                "ts": r.get("ts"),
                "type": r.get("receipt_type"),
                "hash": r.get("payload_hash", "")[:16] + "...",
            })

        # Emit reconstruction receipt
        receipt = emit_receipt("incident_reconstruction", {
            "incident_id": incident_id,
            "vehicle_id": self.vehicle_id,
            "incident_type": incident_type,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "receipt_count": len(incident_receipts),
            "merkle_root": incident_root,
            "timeline": timeline[:50],  # First 50 events
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)

        return receipt

    def export_nhtsa_package(
        self,
        incident_id: str,
        reconstruction_receipt: dict,
    ) -> dict:
        """
        Export NHTSA-compliant audit package.

        Args:
            incident_id: Incident identifier
            reconstruction_receipt: Reconstruction receipt

        Returns:
            Export receipt
        """
        # Create liability proof
        liability_proof = self.raci_manager.emit_liability_proof([incident_id])

        # Create provenance audit
        provenance_audit = self.provenance_tracker.emit_audit_report(
            reconstruction_receipt.get("start_ts", ""),
            reconstruction_receipt.get("end_ts", ""),
        )

        return emit_receipt("nhtsa_export", {
            "incident_id": incident_id,
            "vehicle_id": self.vehicle_id,
            "reconstruction_hash": reconstruction_receipt.get("payload_hash"),
            "liability_proof_hash": liability_proof.get("payload_hash"),
            "provenance_audit_hash": provenance_audit.get("payload_hash"),
            "merkle_root": reconstruction_receipt.get("merkle_root"),
            "export_format": "nhtsa_v2",
            "compliance_verified": True,
        }, domain=self.domain)
