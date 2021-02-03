from configuration.settings import Settings
from runner import Runner


def main(settings=Settings()):
    runner = Runner(settings)
    runner.run()


if __name__ == '__main__':
    main()
