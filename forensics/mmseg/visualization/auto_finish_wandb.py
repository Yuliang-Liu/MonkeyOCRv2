import os
import re
import signal
import threading
from typing import Optional, Dict
from mmengine.visualization import WandbVisBackend

from mmseg.registry import VISBACKENDS
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

@VISBACKENDS.register_module()
class WandbAutoFinishBackend(WandbVisBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def force_finish_wandb(self):
        log_file_path = os.path.join(self._wandb.run.dir + '/../logs/debug-internal.log')
        with open(log_file_path, 'r') as f:
            last_line = f.readlines()[-1]
        match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
        if match:
            pid = int(match.group(2))
            print(f'wandb pid: {pid}')
        else:
            print('Cannot find wandb process-id.')
            return

        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Process with PID {pid} killed successfully.")
        except OSError:
            print(f"Failed to kill process with PID {pid}.")

    def close(self) -> None:
        if hasattr(self, '_wandb'):
            threading.Timer(240, self.force_finish_wandb).start()
            """close an opened wandb object."""
            self._wandb.join()
