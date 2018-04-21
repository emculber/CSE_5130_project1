from __future__ import print_function
import os
import neat
import visualize
import numpy as np
from bridge import Bridge

screen_size_x = 256
screen_size_y = 224
bridge = Bridge()
pelletsMax = 177

def eval_genomes(genomes, config):
    size = len(genomes)
    count = 1
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        done = False
        bridge.reset()
        reward_last = pelletsMax
        while not done:
            outputs = ['up', 'down', 'left', 'right']
            screen = bridge.getPixelScreen().flatten()
            output = net.activate(screen)
            logits_exp = np.exp(output)
            probs = logits_exp / np.sum(logits_exp)
            action = np.random.choice(outputs, p=probs)
            reward, done = bridge.stepScore(action)
            if reward_last - reward < 0:
                reward_last = pelletsMax
            genome.fitness += reward_last - reward
            reward_last = reward
        print(str(count) + "/" + str(size) + " Fitness: " + str(genome.fitness))
        count+=1

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # dont = False
    # for not done:
    #    output = winner_net.activate(bridge.getScreen())
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    visualize.draw_net(config, winner, True, node_names=node_names)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)

if __name__ == '__main__':
    bridge.connectToSocket()
    bridge.reset()
    #bridge.sendAndForget("skip:60")

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
