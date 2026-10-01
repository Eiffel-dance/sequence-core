from app import Linear,TanhSequence
print(TanhSequence(Linear([0.4,0.2])).forward([[1],[2],[3]],2))
