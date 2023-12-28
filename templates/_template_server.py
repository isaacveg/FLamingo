from FLamingo.core.server import *
from _template_network import MyNetworkHandler


class YourClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        # define your own information
        self.network = MyNetworkHandler()
        self.new_client_attr = None

    def do_something_new(self):
        # do something you want
        pass


class YourServer(Server):
    def __init__(self, args):
        super().__init__(args)
        # your server attrs
        self.new_server_attr = None


    def your_select_clients(self):
        # maybe you want your own algorithms created,
        # or just use self.select_clients
        pass
    

    def do_something_new(self):
        pass

    def run(self):
        """
        Basically init client, 
        select, broadcast, listen, aggregate 
        and wait for next round
        """
        self.init_clients(clientObj=YourClientInfo)
        while True:
            self.your_select_clients()
            self.broadcast(data={'status':'TRAINING'})
            self.listen()
            self.aggregate(weight_by_sample=True)
            self.log("log something")
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Yes, just end your job")
        self.stop_all()


if __name__ == "__main__":
    args = get_args()
    server = YourServer(args)
    server.run()