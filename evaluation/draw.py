import pickle
import matplotlib.pyplot as plt

with open('adjectives_balance.pkl', 'rb') as f:
    data = pickle.load(f)

sorted_words = sorted(data.items(), key= lambda k : k[1], reverse=True)
num = 40

x = [s[0] for s in sorted_words[:num]]
y = [s[1] for s in sorted_words[:num]]

x_pos = [i*2 for i, _ in enumerate(x)]
x_pos.reverse()

plt.barh(x_pos, y, color='#004080', height=1.5)
plt.ylabel("Word Frequency")
plt.title("Masked Nouns Frequency")

plt.yticks(x_pos, x)
plt.tight_layout()
plt.savefig('wordfrequency.jpg')
