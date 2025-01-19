import hydra
from openteach.components import TeleOperator
import os
import signal

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'teleop')
def main(configs):
    teleop = TeleOperator(configs)
    processes = teleop.get_processes()

    for process in processes:
        process.start()

    # for process in processes:
    #     process.join()
    while True:
        for process in processes:
            if not process.is_alive():
                for p in processes:
                    if p.is_alive():
                        os.kill(p.pid, signal.SIGINT)  # Send SIGINT to the process
                return 0

if __name__ == '__main__':
    main()