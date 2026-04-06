from blockchain_client import load_config, InsuranceAuditClient

config = load_config()
client = InsuranceAuditClient(config)

claim_id = "CLM-2026-2001"
image_hash = "sha256-test-hash-2001"
predicted_class = "03-severe"
confidence_bps = 8421

tx_hash, receipt = client.submit_claim(
    claim_id=claim_id,
    image_hash=image_hash,
    predicted_class=predicted_class,
    confidence_bps=confidence_bps,
)

print("Submit tx hash:", tx_hash)
print("Receipt status:", receipt["status"])

claim = client.get_claim(claim_id)
print("Stored claim:", claim)

verified = client.verify_image_hash(claim_id, image_hash)
print("Hash verified:", verified)