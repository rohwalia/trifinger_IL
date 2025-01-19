import hydra
from openteach.components import Collector

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'collect_data')
def main(configs):
    print(configs)
    collector = Collector(configs, configs.demo_num, configs.task)
    processes = collector.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()