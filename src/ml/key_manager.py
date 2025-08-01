from datetime import datetime


class KeyManager:
    def __init__(self, api_keys, cooldown_seconds=60):
        self.api_keys = api_keys
        self.cooldown_seconds = cooldown_seconds
        self.cooldowns = {k: 0.0 for k in api_keys}
        self.uses = {k: 0 for k in api_keys}
        self.failures = {k: 0 for k in api_keys}

    def now(self):
        return datetime.now().timestamp()

    def get_key(self):
        valid_keys = [
            (k, self.uses[k], self.failures[k])
            for k in self.api_keys
            if self.cooldowns[k] <= self.now()
        ]
        if not valid_keys:
            return None
        valid_keys.sort(key=lambda x: (x[2], x[1]))  # least failures, then least used
        return valid_keys[0][0]

    def record_429(self, key):
        self.failures[key] += 1
        if self.failures[key] >= 3:
            self.cooldowns[key] = self.now() + self.cooldown_seconds
            self.failures[key] = 0

    def record_success(self, key):
        self.uses[key] += 1
        self.failures[key] = 0
