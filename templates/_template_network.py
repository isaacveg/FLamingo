from FLamingo.core.network import *


class MyNetworkHandler(NetworkHandler):
    def __init__(self, world, rank, size):
        super().__init__(world, rank, size)
        # 在这里可以添加自定义的初始化代码

    def send_request(self, request):
        # 在这里可以添加自定义的请求发送逻辑
        # 返回响应数据
        return super().send_request(request)