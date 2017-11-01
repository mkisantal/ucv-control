from command import Commander
import matplotlib.pyplot as plt
import yaml


# note that UE4 uses left-handed coordinate system, but obstacles are stored in right-handed coordinate system
# unit: cm

filename = './map.yaml'

class Mapper:
    def __init__(self, start_sim=True):
        if start_sim:
            self.cmd = Commander(0)
        else:
            self.cmd = None

        self.obstacles = []

    def map(self, xmin, xmax, ymin, ymax, stepsize=50, height=150, save=False):

        print('Checking collisions at {} locations.'.format((xmax-xmin)*(ymax-ymin)/(stepsize * stepsize)))
        self.cmd.request('vset /camera/0/rotation {:.2f} {:.2f} {:.2f}'.format(0, 0, 0))
        for x in range(xmin, xmax, stepsize):
            for y in range(ymin, ymax, stepsize):
                self.cmd.request('vset /camera/0/location {:.2f} {:.2f} {:.2f}'.format(y, x, height))
                self.cmd.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(y+10, x, height))
                final_loc = [float(v) for v in self.cmd.request('vget /camera/0/location').split(' ')]
                if final_loc != [y+10, x, height]:
                    self.obstacles.append([y, x, height])

        if save:
            with open(filename, 'w') as map_file:
                yaml.dump(self.obstacles, stream=map_file, default_flow_style=False)

    def plot(self, axis):
        for obstacle in self.obstacles:
            axis.plot(obstacle[0], obstacle[1], 'ks')

if __name__ == '__main__':

    load = True
    mapper = Mapper(start_sim=(not load))
    xmin = 0
    xmax = 1000
    ymin = 0
    ymax = 1000


    if load:
        with open(filename, 'r') as map_file:
            mapper.obstacles = yaml.load(map_file)
    else:
        mapper.map(0, 1000, 0, 1000, save=True)
        mapper.cmd.shut_down()

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(xmin-100, xmax+100)
    ax.set_ylim(ymin-100, ymax+100)
    plt.ion()
    plt.show()

    mapper.plot(ax)

    plt.pause(30)



