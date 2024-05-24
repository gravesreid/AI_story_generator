class character:
    def __init__(self, name, description, gender):
        self.name = name
        self.description = description
        self.gender = gender

    def __repr__(self):
        return f"{self.name} is {self.description}."
    
    def get_description(self):
        return self.description
    
    def get_name(self):
        return self.name
    
    def get_gender(self):
        return self.gender

fluffy = character("Fluffy", "A curious and adventurous squirrel with a fluffy tail", "Female")
thumper = character("Thumper", "A playful and energetic rabbit with a knack for finding hidden paths", "Male")
honey = character("Honey", "A sweet and nurturing bear who loves to help others", "Female")
rocky = character("Rocky", "A brave and strong raccoon who is always ready for a challenge", "Male")
twinkle = character("Twinkle", "A wise and gentle owl who knows the secrets of the forest", "Female")
dash = character("Dash", "A swift and clever fox who loves solving puzzles and riddles", "Male")
