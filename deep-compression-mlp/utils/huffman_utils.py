def apply_huffman_to_npz(npz_path):
    import numpy as np
    from collections import Counter
    from compression.huffman import build_huffman_tree

    data = np.load(npz_path)

    print("\n[*] Applying Huffman Encoding...\n")

    all_values = []

    for key in data:
        all_values.extend(data[key].flatten())

    freq = Counter(all_values)
    codebook = build_huffman_tree(freq)

    print("   --- HUFFMAN DICTIONARY (Snippet) ---")
    print(f"   [-] Unique Weight Clusters: {len(freq)}")

    # print top 5 frequent
    top = sorted(freq.items(), key=lambda x: -x[1])[:5]

    for val, f in top:
        code = codebook[val]
        print(f"   [-] Value: {val:>8.4f} | Freq: {f:>7} | Huffman Code: {code} ({len(code)} bits)")

    print("   [-] ... (and so on)")
    print("   ------------------------------------")

    # compute compression
    total_bits = sum(freq[v] * len(codebook[v]) for v in freq)
    original_bits = len(all_values) * 32

    ratio = original_bits / total_bits

    return ratio, total_bits / 8 / (1024*1024)