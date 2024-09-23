#Object Oriented Programming

#Create the class
class Player:
    def __init__(self, name, pos, catches, targets, rushing_attempts=None, rushing_yds=None):
        self.name = name
        self.pos = pos
        self.catches = catches
        self.targets = targets
        self.rushing_attempts = rushing_attempts
        self.rushing_yds = rushing_yds

    def catch_rate(self):
        return self.catches/self.targets

    def yards_per_carry(self):
        return self.rushing_yds/self.rushing_attempts

    def efficiency(self):
        return {
            'yards_per_carry': self.yards_per_carry(),
            'catch_rate': self.catch_rate()
        }

#creating instances of a class
jj = Player(name='Justin Jefferson', pos='WR', catches=128, targets=184)
ceedee = Player(name='CeeDee Lamb', pos='WR', catches=128, targets=156, rushing_attempts=10, rushing_yds=82)

#call our catch_rate method and attributes
print(ceedee.catches)
print(ceedee.catch_rate())
print(ceedee.yards_per_carry())
print(ceedee.efficiency())

