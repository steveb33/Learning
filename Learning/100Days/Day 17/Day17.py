class User:
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username
        self.followers = 0
        self.following = 0

    def follow(self, user):
        user.followers += 1
        self.following += 1

user_1 = User('001', 'Steve')
user_2 = User('001', 'Random')

user_1.follow(user_2)

print(user_1.following)
print(user_2.followers)

"""Not adding the course code used for the rest of this day and future days because I don't think I'm allowed to"""