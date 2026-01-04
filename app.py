import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def run():
    st.title("🕸️ Vaka 5: Örümcek Ağı")
    
    if 'math_mode_5' not in st.session_state: st.session_state['math_mode_5'] = False
    st.markdown("**Görev:** İlişkileri analiz et. Casusları bul." if not st.session_state['math_mode_5'] else "### 📐 GNN Message Passing")

    with st.sidebar:
        n = st.slider("Kişi", 10, 50, 20)
        iters = st.slider("Tur (Epoch)", 0, 10, 0)
        self_w = st.slider("Öz İrade (Self-Weight)", 0.0, 1.0, 0.2)
        if st.button("Sıfırla"): st.session_state['G5'] = None

    if 'G5' not in st.session_state or st.session_state['G5'] is None:
        G = nx.watts_strogatz_graph(n, 4, 0.2, seed=42)
        val = {node: 0.5 for node in G.nodes()}
        fixed = [0, n-1]
        val[0] = 1.0; val[n-1] = 0.0
        st.session_state['G5'] = G
        st.session_state['val5'] = val
        st.session_state['fix5'] = fixed

    G = st.session_state['G5']
    curr = st.session_state['val5'].copy()
    
    for _ in range(iters):
        new_val = curr.copy()
        for node in G.nodes():
            if node in st.session_state['fix5']: continue
            neigh = list(G.neighbors(node))
            avg = sum(curr[x] for x in neigh) / len(neigh)
            new_val[node] = (self_w * curr[node]) + ((1-self_w) * avg)
        curr = new_val

    fig, ax = plt.subplots()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, node_color=[curr[n] for n in G.nodes()], cmap=plt.cm.coolwarm, vmin=0, vmax=1)
    st.pyplot(fig)
    
    st.divider()
    if st.button("🔴 Kırmızı Hap"):
        st.session_state['math_mode_5'] = not st.session_state['math_mode_5']
        if hasattr(st, "rerun"): st.rerun() 
        else: st.experimental_rerun()

if __name__ == "__main__":
    run()