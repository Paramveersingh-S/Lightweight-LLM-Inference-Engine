class BatchScheduler:
    """
    Manages continuous batching for incoming requests.
    """
    def __init__(self, kv_cache, config):
        self.kv_cache = kv_cache
        self.config = config
        self.queue = []
        
    def add_requests(self, requests):
        self.queue.extend(requests)
        
    def next_batch(self):
        return self.queue.pop(0) if self.queue else None
        
    def all_done(self):
        return len(self.queue) == 0
        
    def update(self, batch, tokens):
        pass
