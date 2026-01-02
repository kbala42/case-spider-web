import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Vaka: Örümcek Ağı (GNN)", layout="wide")

st.title("🕸️ Vaka: Örümcek Ağı (Kolektif Zeka - GNN)")
st.markdown("""
**Sherlock'un Notu:** "Tek bir suçluyu yakalamak kolaydır. Ama suç, bir virüs gibi ilişkiler üzerinden yayılır. 
Burada tek tek insanları incelemeyeceğiz. **İlişkilerin kendisine** bakarak kimin kim olduğunu anlayacağız."

**Teknoloji:** Graph Neural Networks (GNN). Google Haritalar trafiği böyle tahmin eder, Biyologlar ilaçları böyle keşfeder.
**Mennan Usta Prensibi:** "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim."
""")

# --- YAN PANEL: AĞ AYARLARI ---
with st.sidebar:
    st.header("🕸️ Ağ Kurulumu")
    num_nodes = st.slider("Kişi Sayısı", 10, 50, 30)
    connection_prob = st.slider("Bağlantı Sıklığı", 0.1, 0.5, 0.15)
    
    st.divider()
    st.header("🧠 GNN Motoru")
    iterations = st.slider("Mesajlaşma Turu (Epochs)", 1, 10, 1)
    
    if st.button("Ağı Yeniden Kur"):
        st.session_state['gnn_graph'] = None

# --- GNN FONKSİYONLARI ---

def init_graph(n, p):
    # Rastgele bir ağ oluştur
    G = nx.watts_strogatz_graph(n, k=4, p=p)
    
    # Herkese başlangıçta "Bilinmiyor" (0.5) değeri ver
    # 0.0 = Kesin MAVİ (Sivil)
    # 1.0 = Kesin KIRMIZI (Casus)
    # 0.5 = GRİ (Bilmiyoruz)
    values = {node: 0.5 for node in G.nodes()}
    
    # Şüphe Tohumlarını Ek (Labels)
    # Rastgele 2 kişiyi seç: Biri kesin Casus, biri kesin Sivil
    spies = [0] # 0. düğüm CASUS olsun
    civilians = [n-1] # Son düğüm SİVİL olsun
    
    values[0] = 1.0   # Kırmızı
    values[n-1] = 0.0 # Mavi
    
    return G, values, spies, civilians

def message_passing(G, values, fixed_nodes):
    # GNN'in Kalbi: Komşulardan Bilgi Topla
    new_values = values.copy()
    
    for node in G.nodes():
        if node in fixed_nodes:
            continue # Tohumların fikri değişmez (Onlar kanıtlanmış suçlu/sivil)
            
        # Komşuları bul
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
            
        # Komşuların değerlerinin ortalamasını al
        neighbor_sum = sum([values[n] for n in neighbors])
        neighbor_avg = neighbor_sum / len(neighbors)
        
        # Basit GNN Formülü: Kendi fikrimle komşularımın fikrini harmanla
        # %20 Kendi fikrim, %80 Çevre etkisi (Uyum sağlama)
        new_values[node] = (0.2 * values[node]) + (0.8 * neighbor_avg)
        
    return new_values

# --- ANA AKIŞ ---

if 'gnn_graph' not in st.session_state or st.session_state['gnn_graph'] is None:
    G, val, spies, civs = init_graph(num_nodes, connection_prob)
    st.session_state['gnn_graph'] = G
    st.session_state['node_values'] = val
    st.session_state['fixed_nodes'] = spies + civs

G = st.session_state['gnn_graph']
values = st.session_state['node_values']
fixed_nodes = st.session_state['fixed_nodes']

# Görselleştirme Paneli
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Analiz Sahası (Tur: {iterations})")
    
    # GNN Algoritmasını Çalıştır (Seçilen tur kadar)
    current_values = values.copy()
    for _ in range(iterations):
        current_values = message_passing(G, current_values, fixed_nodes)
    
    # Çizim
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.kamada_kawai_layout(G) # Estetik bir düzen
    
    # Düğümleri Renklendir (Değerlerine göre Mavi-Gri-Kırmızı skalası)
    node_colors = [current_values[n] for n in G.nodes()]
    
    # Düğümleri Çiz
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.coolwarm, 
                                   node_size=500, vmin=0, vmax=1, edgecolors='black')
    
    # Kenarları Çiz
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Etiketler
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=10)
    
    # Renk Barı (Skala)
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("0 (Sivil) <----> 1 (Casus)")
    
    st.pyplot(fig)

with col2:
    st.subheader("📊 Ağ Raporu")
    
    # İstatistikler
    spy_count = sum(1 for v in current_values.values() if v > 0.7)
    civ_count = sum(1 for v in current_values.values() if v < 0.3)
    uncertain = num_nodes - spy_count - civ_count
    
    st.metric("Tespit Edilen Casuslar", f"{spy_count} Kişi")
    st.metric("Güvenli Siviller", f"{civ_count} Kişi")
    st.metric("Hala Şüpheli (Gri)", f"{uncertain} Kişi")
    
    st.info("""
    **Nasıl Çalıştı?**
    Başlangıçta sadece 2 kişi renkliydi. 'Mesajlaşma' turlarını artırdıkça, 
    bilgi ağ üzerinden yayıldı ve gri düğümler komşularının rengine büründü.
    """)

    with st.expander("👨‍🏫 Mennan Usta ve GNN"):
        st.write("""
        "Evlat, bu sistemin aynısı sanayide de vardır. 
        Bir atölyede iki tane tembel usta varsa, yanlarına kimi koyarsan koy zamanla o da yavaşlar.
        İki tane çalışkan varsa, çırağı da çalışkan yaparlar.
        
        Bilgisayarcılar buna 'Graph Convolution' diyor, biz 'Ortamın Hali' diyoruz."
        """)
