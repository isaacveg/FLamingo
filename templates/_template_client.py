from FLamingo.core.client import *

class YourClient(Client):
    """
    Your own Client
    """
    def __init__(self, args):
        super().__init__(args)
        # your new attributes
        self.new_attr = None

    def YourTrain(self):
        # Train your own model
        return 

    def run(self):
        """
        Client jobs, usually have a loop
        """
        while True:
            # get from server
            data = self.network.get(0)
            if data['status'] == 'TRAINING':
                print('training...')
                # Do something
            
            elif data['status'] == 'STOP':
                print('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        print('stopped')


if __name__ == '__main__':
    args = get_args()
    client = YourClient(args)
    client.run()