class ScheduledVariable:
    def __init__(self, start_value, end_value, start_step, end_step):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

        self.decay_rate = end_value/start_value

    def interpolate(self, step):
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        else:
            res = self.start_value +\
                  (self.end_value - self.start_value) * ((step-self.start_step)/(self.end_step - self.start_step))
            return res

    def exponential_decay(self, step):
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        else:
            res = self.start_value * self.decay_rate**((step-self.start_step)/(self.end_step - self.start_step))
            return res


class Schedule:
    def __init__(self, config):
        # learning rate
        self._learning_rate = ScheduledVariable(config.LEARNING_RATE.START_VALUE,
                                                config.LEARNING_RATE.END_VALUE,
                                                config.LEARNING_RATE.START_STEP,
                                                config.LEARNING_RATE.END_STEP)
        self._entropy = ScheduledVariable(config.ENTROPY.START_VALUE,
                                          config.ENTROPY.END_VALUE,
                                          config.ENTROPY.START_STEP,
                                          config.ENTROPY.END_STEP)

    def learning_rate(self, step):
        # return self._learning_rate.interpolate(step)
        return self._learning_rate.exponential_decay(step)

    def entropy(self, step):
        return self._entropy.interpolate(step)





