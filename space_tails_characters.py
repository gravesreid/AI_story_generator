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

astro = character("Astro", "A brilliant young astronaut with a love for exploring new planets", "Male")
nova = character("Nova", "A tech-savvy engineer who creates innovative gadgets for space travel", "Female")
zeno = character("Zeno", "An alien from the planet Zygor with a knack for solving cosmic puzzles", "Male")
stella = character("Stella", "A fearless pilot who can navigate through the trickiest asteroid fields", "Female")
quasar = character("Quasar", "A wise and ancient robot with a vast knowledge of the universe", "Male")
luna = character("Luna", "A curious and adventurous space scientist always ready for new discoveries", "Female")
