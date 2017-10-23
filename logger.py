import time


class TestLogger:
    def __init__(self, workers):
        self.instances = len(workers)
        self.should_stop = False
        self.workers = workers

    def work(self):
        print('[logger] Starting Logger')
        with open('sim_test_log', 'w') as log_file:
            log_file.write('Simulator instances: {}\n'.format(self.instances))
            while not self.should_stop:
                time.sleep(10)
                crashed = 0
                hour = time.localtime().tm_hour
                minute = time.localtime().tm_min
                for worker in self.workers:
                    if not worker.env.client.isconnected():
                        crashed += 1
                if crashed is 0:
                    log_file.write('{}:{}, {}\n'.format(hour, minute, crashed))
                else:
                    log_file.write('{}:{}, {}\n'.format(hour, minute, crashed))
            print('[logger] Shutting down.')


class CumulativeStepsLogger:
    def __init__(self):
        self.counter = 0
        self.d = 0
        self.should_stop = False

    def increment(self):
        self.counter += 1

    def work(self):
        with open('step_count_log', 'w') as log_file:
            while not self.should_stop:
                time.sleep(10)
                diff = self.counter - self.d
                self.d = self.counter
                hour = time.localtime().tm_hour
                minute = time.localtime().tm_min
                log_file.write('{}:{}, {}, {}\n'.format(hour, minute, self.counter, diff))



