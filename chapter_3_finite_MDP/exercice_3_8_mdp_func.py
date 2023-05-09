
if __name__ == "__main__":

	R = [-1, 2, 6, 3, 2]
	G = [0] * (len(R) + 1)

	gamma = 0.5

	for i in range(len(R) - 1, -1, -1):
		G[i] = R[i] + gamma * G[i + 1]

	print(G)
