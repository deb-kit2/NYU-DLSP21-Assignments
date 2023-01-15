import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_feature s: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x) :
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function

        self.cache["x"] = x

        s1 = torch.mm(x, torch.transpose(self.parameters["W1"], 1, 0))
        s1 = s1 + self.parameters["b1"].expand_as(s1)
        # self.cache["s1+b"] = s1
        self.cache["s1"] = s1

        if self.f_function == "relu" :
            s2 = torch.nn.functional.relu(s1)
        elif self.f_function == "sigmoid" :
            s2 = torch.nn.functional.sigmoid(s1)
        self.cache["s2"] = s2

        s3 = torch.mm(s2, torch.transpose(self.parameters["W2"], 1, 0))
        s3 = s3 + self.parameters["b2"].expand_as(s3)
        self.cache["s3"] = s3
        # self.cache["s3+b"] = s3

        y_hat = s3
        if self.g_function == "relu" :
            y_hat = torch.nn.functional.relu(s3)
        elif self.g_function == "sigmoid" :
            y_hat = torch.nn.functional.sigmoid(s3)
        self.cache["y_hat"] = y_hat

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function

        dJds3 = dJdy_hat
        if self.g_function == "sigmoid" :
            dJds3 = self.cache["y_hat"] * (1 - self.cache["y_hat"]) * dJdy_hat
        elif self.g_function == "relu" :
            dJds3 = (self.cache["s3"] > 0) * dJdy_hat
        # dJds3 : N, l2out
        
        dJdW2 = torch.mm(torch.transpose(dJds3, 1, 0), self.cache["s2"])
        dJdb2 = torch.sum(dJds3, dim = 0)
        # dJdW2 : l2out, l2in :: dJdb2 : l2out

        dJds2 = torch.mm(dJds3, self.parameters["W2"])
        if self.f_function == "sigmoid" :
            dJds1 = self.cache["s2"] * (1 - self.cache["s2"]) * dJds2
        elif self.f_function == "relu" :
            dJds1 = (self.cache["s1"] > 0) * dJds2
        # dJsd2 = N, l2in = N, l1out

        dJdW1 = torch.mm(torch.transpose(dJds1, 1, 0), self.cache["x"])
        dJdb1 = torch.sum(dJds1, dim = 0)
        # dJdW2 : l2out, l2in :: dJdb2 : l2out
        
        self.grads["dJdW1"] = dJdW1
        self.grads["dJdb1"] = dJdb1
        self.grads["dJdW2"] = dJdW2
        self.grads["dJdb2"] = dJdb2

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

# TODO
def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.mean((y - y_hat)**2)

    n, m = tuple(y.size())
    dJdy_hat = (y_hat - y) * 2 / (n * m)

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)
    loss = -torch.mean(loss)

    n, m = tuple(y.size())
    dJdy_hat = y * torch.reciprocal(y_hat) - (1 - y) * torch.reciprocal(1 - y_hat)
    dJdy_hat = -dJdy_hat / (n * m)

    return loss, dJdy_hat
