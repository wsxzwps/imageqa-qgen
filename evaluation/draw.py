import pickle
import matplotlib.pyplot as plt

with open('nouns.pkl', 'rb') as f:
    data = pickle.load(f)

sorted_words = sorted(data.items(), key= lambda k : (k[1], k[0]), reverse=True)
num = 40

x = [s[0] for s in sorted_words[:num]]
y = [s[1] for s in sorted_words[:num]]

x_pos = [i for i, _ in enumerate(x)]

plt.barh(x_pos, y, color='#ff8c66')
plt.ylabel("Word Frequency")
plt.title("Masked Nouns Frequency")

plt.yticks(x_pos.reverse(), x)
plt.savefig('wordfrequency.jpg')
