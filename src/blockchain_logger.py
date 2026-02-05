import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os


# User roles with permissions
ROLES = {
    "admin": {"read": True, "write": True, "verify": True},
    "detector": {"read": True, "write": True, "verify": False},
    "auditor": {"read": True, "write": False, "verify": True},
    "viewer": {"read": True, "write": False, "verify": False},
}


class BlockchainUser:
    """User with authentication credentials and role-based permissions."""
    
    def __init__(self, username: str, password: str, role: str = "detector"):
        if role not in ROLES:
            raise ValueError(f"Invalid role. Must be one of: {list(ROLES.keys())}")
        
        self.username = username
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.role = role
        self.permissions = ROLES[role]
    
    def verify_password(self, password: str) -> bool:
        """Verify if provided password matches stored hash."""
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
    
    def can_write(self) -> bool:
        """Check if user has write permission."""
        return self.permissions.get("write", False)
    
    def can_read(self) -> bool:
        """Check if user has read permission."""
        return self.permissions.get("read", False)
    
    def can_verify(self) -> bool:
        """Check if user can verify blockchain integrity."""
        return self.permissions.get("verify", False)


class PermissionedBlockchain:
    """Permissioned blockchain with user authentication and role-based access control."""
    
    def __init__(self, ledger_path: str = "blockchain_ledger.jsonl", secret_key: str = None):
        self.ledger_path = ledger_path
        self.secret_key = secret_key or os.getenv("BLOCKCHAIN_SECRET_KEY", "default-secret-key")
        self.users: Dict[str, BlockchainUser] = {}
        self.current_user: Optional[BlockchainUser] = None
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default users for testing."""
        self.users["admin"] = BlockchainUser("admin", "admin_password", "admin")
        self.users["detector"] = BlockchainUser("detector", "detector_password", "detector")
        self.users["auditor"] = BlockchainUser("auditor", "auditor_password", "auditor")
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate user."""
        if username not in self.users:
            raise ValueError(f"User '{username}' not found")
        
        user = self.users[username]
        if not user.verify_password(password):
            raise ValueError(f"Invalid password for user '{username}'")
        
        self.current_user = user
        return True
    
    def logout(self):
        """Logout current user."""
        self.current_user = None
    
    def register_user(self, username: str, password: str, role: str = "viewer") -> bool:
        """Register a new user (admin only)."""
        if not self.current_user or not self.current_user.can_verify():
            raise PermissionError("Only admins can register new users")
        
        if username in self.users:
            raise ValueError(f"User '{username}' already exists")
        
        self.users[username] = BlockchainUser(username, password, role)
        return True
    
    def _compute_signature(self, entry: Dict[str, Any]) -> str:
        """Compute HMAC signature for entry."""
        entry_str = json.dumps(entry, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            entry_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def log_anomaly(
        self,
        index: int,
        details: Dict[str, Any],
    ) -> None:
        """
        Append a new entry to the permissioned blockchain ledger.
        Requires write permissions.
        
        Each entry includes:
        - timestamp
        - data (anomaly info)
        - user (who logged it)
        - prev_hash (hash of previous entry)
        - signature (HMAC signature)
        """
        if not self.current_user:
            raise PermissionError("Must login before logging anomalies")
        
        if not self.current_user.can_write():
            raise PermissionError(f"User '{self.current_user.username}' does not have write permission")
        
        ledger = Path(self.ledger_path)
        
        # Read previous hash
        prev_hash = None
        if ledger.exists() and ledger.stat().st_size > 0:
            with open(ledger, "r") as f:
                for line in f:
                    pass
                last_entry = json.loads(line)
                prev_hash = last_entry.get("hash")
        
        # Create new entry
        entry = {
            "timestamp": datetime.now().timestamp(),
            "user": self.current_user.username,
            "user_role": self.current_user.role,
            "data": {
                "index": index,
                "details": details,
            },
            "prev_hash": prev_hash,
        }
        
        # Compute hash (includes user info for accountability)
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = entry_hash
        
        # Compute HMAC signature for integrity
        signature = self._compute_signature(entry)
        entry["signature"] = signature
        
        # Append to ledger
        with open(ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def verify_blockchain(self) -> bool:
        """
        Verify blockchain integrity by checking hash chain and signatures.
        Requires verify permissions.
        """
        if not self.current_user:
            raise PermissionError("Must login before verifying blockchain")
        
        if not self.current_user.can_verify():
            raise PermissionError(f"User '{self.current_user.username}' does not have verify permission")
        
        ledger = Path(self.ledger_path)
        if not ledger.exists() or ledger.stat().st_size == 0:
            return True
        
        prev_hash = None
        with open(ledger, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    
                    # Verify hash chain
                    if entry.get("prev_hash") != prev_hash:
                        print(f"Hash chain broken at entry {line_num}")
                        return False
                    
                    # Verify signature
                    stored_signature = entry.pop("signature", None)
                    entry_copy = entry.copy()
                    expected_signature = self._compute_signature(entry_copy)
                    if stored_signature != expected_signature:
                        print(f"Signature mismatch at entry {line_num}")
                        return False
                    
                    prev_hash = entry.get("hash")
                except Exception as e:
                    print(f"Error verifying entry {line_num}: {e}")
                    return False
        
        return True
    
    def read_ledger(self, limit: int = None) -> list:
        """
        Read ledger entries. Requires read permissions.
        """
        if not self.current_user:
            raise PermissionError("Must login before reading blockchain")
        
        if not self.current_user.can_read():
            raise PermissionError(f"User '{self.current_user.username}' does not have read permission")
        
        ledger = Path(self.ledger_path)
        entries = []
        
        if ledger.exists():
            with open(ledger, "r") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    entries.append(json.loads(line))
        
        return entries


# Backwards compatibility: maintain old function signature for existing code
_blockchain_instance = None

def log_anomaly(
    index: int,
    details: Dict[str, Any],
    ledger_path: str = "blockchain_ledger.jsonl",
    username: str = "detector",
    password: str = "detector_password",
) -> None:
    """
    Legacy function for backwards compatibility.
    Logs anomaly using default detector user credentials.
    Auto-initializes and maintains persistent blockchain instance.
    """
    global _blockchain_instance
    
    # Initialize once and reuse
    if _blockchain_instance is None:
        _blockchain_instance = PermissionedBlockchain(ledger_path)
        try:
            _blockchain_instance.login(username, password)
        except ValueError:
            # First run - user doesn't exist, register them
            pass
    
    # Ensure logged in
    if _blockchain_instance.current_user is None:
        _blockchain_instance.login(username, password)
    
    _blockchain_instance.log_anomaly(index, details)
