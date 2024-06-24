class Vodscillator:
  def __init__(self, **kwargs):
    self.num_nodes = kwargs["num_nodes"]
    self.name = kwargs["name"]

  def __str__(self):
    return f"A vodscillator named {self.name} with {self.num_nodes} nodes!"