import math
class Linear:
    def __init__(self,weight,bias=0.0): self.weight=list(weight); self.bias=bias; self.grad=[0.0]*len(self.weight)
    def forward(self,x): self.last=x; return sum(w*a for w,a in zip(self.weight,x))+self.bias
    def backward(self,grad): self.grad=[grad*a for a in self.last]; return [grad*w for w in self.weight]
    def zero_grad(self): self.grad=[0.0]*len(self.weight)
class TanhSequence:
    def __init__(self,linear): self.linear=linear
    def forward(self,rows,truncate=None):
        self.outputs=[]; self.hidden=0.0
        for i,row in enumerate(rows):
            if truncate and i%truncate==0: self.hidden=0.0
            self.hidden=math.tanh(self.linear.forward([row[0],self.hidden])); self.outputs.append(self.hidden)
        return self.outputs
