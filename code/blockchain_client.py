import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from web3 import Web3


load_dotenv()


@dataclass
class BlockchainConfig:
    rpc_url: str
    private_key: str
    contract_address: str
    chain_id: int


CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "admin", "type": "address"},
            {"internalType": "address", "name": "insurer", "type": "address"},
            {"internalType": "address", "name": "assessor", "type": "address"},
        ],
        "stateMutability": "nonpayable",
        "type": "constructor",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "role", "type": "bytes32"}],
        "name": "AccessControlBadConfirmation",
        "type": "error",
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "AccessControlUnauthorizedAccount",
        "type": "error",
    },
    {"inputs": [], "name": "EnforcedPause", "type": "error"},
    {"inputs": [], "name": "ExpectedPause", "type": "error"},
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "claimKey", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "claimId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "imageHash", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "predictedClass", "type": "string"},
            {"indexed": False, "internalType": "uint16", "name": "confidenceBps", "type": "uint16"},
            {"indexed": True, "internalType": "address", "name": "submittedBy", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "createdAt", "type": "uint256"},
        ],
        "name": "ClaimSubmitted",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "claimKey", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "claimId", "type": "string"},
            {"indexed": False, "internalType": "uint8", "name": "oldStatus", "type": "uint8"},
            {"indexed": False, "internalType": "uint8", "name": "newStatus", "type": "uint8"},
            {"indexed": True, "internalType": "address", "name": "updatedBy", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "updatedAt", "type": "uint256"},
        ],
        "name": "ClaimStatusUpdated",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "ASSESSOR_ROLE",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "DEFAULT_ADMIN_ROLE",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "INSURER_ROLE",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "claimId", "type": "string"}],
        "name": "claimExists",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "claimId", "type": "string"}],
        "name": "getClaim",
        "outputs": [
            {"internalType": "bool", "name": "exists", "type": "bool"},
            {"internalType": "string", "name": "outClaimId", "type": "string"},
            {"internalType": "string", "name": "outImageHash", "type": "string"},
            {"internalType": "string", "name": "outPredictedClass", "type": "string"},
            {"internalType": "uint16", "name": "outConfidenceBps", "type": "uint16"},
            {"internalType": "uint8", "name": "outStatus", "type": "uint8"},
            {"internalType": "uint256", "name": "outCreatedAt", "type": "uint256"},
            {"internalType": "uint256", "name": "outUpdatedAt", "type": "uint256"},
            {"internalType": "address", "name": "outSubmittedBy", "type": "address"},
            {"internalType": "address", "name": "outUpdatedBy", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "pause",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "paused",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "unpause",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "claimId", "type": "string"},
            {"internalType": "string", "name": "imageHash", "type": "string"},
            {"internalType": "string", "name": "predictedClass", "type": "string"},
            {"internalType": "uint16", "name": "confidenceBps", "type": "uint16"},
        ],
        "name": "submitClaim",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "claimId", "type": "string"},
            {"internalType": "uint8", "name": "newStatus", "type": "uint8"},
        ],
        "name": "updateClaimStatus",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "claimId", "type": "string"},
            {"internalType": "string", "name": "imageHash", "type": "string"},
        ],
        "name": "verifyImageHash",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def load_config() -> BlockchainConfig:
    rpc_url = os.getenv("SEPOLIA_RPC_URL", "")
    private_key = os.getenv("PRIVATE_KEY", "")
    contract_address = os.getenv("CONTRACT_ADDRESS", "")
    chain_id = int(os.getenv("CHAIN_ID", "11155111"))

    if not rpc_url:
        raise ValueError("Missing SEPOLIA_RPC_URL in .env")
    if not private_key:
        raise ValueError("Missing PRIVATE_KEY in .env")
    if not contract_address:
        raise ValueError("Missing CONTRACT_ADDRESS in .env")

    return BlockchainConfig(
        rpc_url=rpc_url,
        private_key=private_key,
        contract_address=contract_address,
        chain_id=chain_id,
    )


class InsuranceAuditClient:
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Sepolia RPC")

        self.account = self.w3.eth.account.from_key(config.private_key)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.contract_address),
            abi=CONTRACT_ABI,
        )

    def _build_base_tx(self) -> Dict[str, Any]:
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        latest_block = self.w3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas", self.w3.eth.gas_price)
        max_priority_fee = self.w3.to_wei(2, "gwei")
        max_fee = base_fee + max_priority_fee * 2

        return {
            "from": self.account.address,
            "nonce": nonce,
            "chainId": self.config.chain_id,
            "maxPriorityFeePerGas": max_priority_fee,
            "maxFeePerGas": max_fee,
        }

    def submit_claim(
        self,
        claim_id: str,
        image_hash: str,
        predicted_class: str,
        confidence_bps: int,
    ) -> Tuple[str, Dict[str, Any]]:
        tx = self.contract.functions.submitClaim(
            claim_id,
            image_hash,
            predicted_class,
            confidence_bps,
        ).build_transaction(self._build_base_tx())

        gas_estimate = self.w3.eth.estimate_gas(tx)
        tx["gas"] = int(gas_estimate * 1.2)

        signed_tx = self.w3.eth.account.sign_transaction(tx, self.config.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_hash.hex(), dict(receipt)

    def get_claim(self, claim_id: str):
        return self.contract.functions.getClaim(claim_id).call()

    def verify_image_hash(self, claim_id: str, image_hash: str) -> bool:
        return self.contract.functions.verifyImageHash(claim_id, image_hash).call()

    def update_claim_status(self, claim_id: str, new_status: int) -> Tuple[str, Dict[str, Any]]:
        tx = self.contract.functions.updateClaimStatus(
            claim_id,
            new_status,
        ).build_transaction(self._build_base_tx())

        gas_estimate = self.w3.eth.estimate_gas(tx)
        tx["gas"] = int(gas_estimate * 1.2)

        signed_tx = self.w3.eth.account.sign_transaction(tx, self.config.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_hash.hex(), dict(receipt)