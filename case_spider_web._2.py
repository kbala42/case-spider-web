import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def run():
    st.title("🕸️ Vaka 5: Örümcek Ağı (Kolektif Zeka - GNN)")

    # --- 1. BAĞLANTI KONTROLÜ (Story Arc) ---
    # Vaka 4'teki Nöron eğitimini tamamlamış olması lazım.
    # (Not: Test ederken hata almamak için geçici bir 'bypass' ekliyorum ama normalde kilitli olmalı)
    if 'train_neuron' not in st.session_state:
        st.warning("⚠️ UYARI: Dedektif, normalde önce Vaka 4'ü (Nöron) tamamlaman gerekirdi. Şimdilik sistemin kilidini 'Acil Durum' koduyla açıyoruz.")
    else:
        st.success("✅ Yetki Onaylandı: Nöral Ağ Mimarisi Aktif.")

    # --- 2. HİKAYE / MATEMATİK MODU ---
    if 'math_mode_5' not in st.session_state:
        st.session_state['math_mode_5'] = False

    if not st.session_state['math_mode_5']:
        st.markdown("""
        **Görev:** Moriarty tek bir kişi değil, bir **AĞ**. 
        Londra yeraltı dünyasında kimin casus olduğunu tek tek bulamayız.
        Ama **ilişkileri** analiz ederek, masum görünenlerin aslında kime hizmet ettiğini bulacağız.
        
        **Mennan Usta Prensibi:** "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim." (Üzüm üzüme baka baka kararır).
        """)
    else:
        st.markdown("""
        ### 📐 MATEMATİKSEL YÜZLEŞME
        **Konu:** Graph Convolutional Networks (GCN) - Mesajlaşma
        
        "Arkadaş etkisi" dediğimiz şey, matematikte **Komşuluk Matrisi ile Durum Vektörünün Çarpımıdır**:
        
        $$ H^{(k+1)} = \sigma( D^{-1} A H^{(k)} W ) $$
        
        * $A$: Komşuluk Matrisi (Kim kiminle bağlı?).
        * $H$: İnsanların mevcut durumu (Casus mu Sivil mi?).
        * Bu formül, her düğümü komşularının ortalamasına çeker (Smoothing).
        """)

    # --- YAN PANEL: AĞ AYARLARI ---
    with st.sidebar:
        st.header("🕸️ Ağ Laboratuvarı")
        num_nodes = st.slider("Kişi Sayısı", 10, 60, 30)
        connection_prob = st.slider("Bağlantı Sıklığı", 0.1, 0.4, 0.15)
        
        st.divider()
        st.header("🧠 GNN Parametreleri")
        iterations = st.slider("Mesajlaşma Turu (Epochs)", 0, 10, 0)
        self_weight = st.slider("Öz İrade (Kendini Koruma)", 0.0, 1.0, 0.2, help="Kişi kendi fikrini ne kadar koruyor?")
        neighbor_weight = 1.0 - self_weight
        st.caption(f"Çevre Etkisi: {neighbor_weight:.1f}")

        if st.button("Ağı Sıfırla / Yeniden Kur"):
            st.session_state['gnn_graph'] = None

    # --- GNN FONKSİYONLARI ---
    def init_graph(n, p):
        G = nx.watts_strogatz_graph(n, k=4, p=p, seed=42)
        # Başlangıç Değerleri (0.5 = Bilinmiyor)
        values = {node: 0.5 for node in G.nodes()}
        
        # Tohumlar (Kesin Bilgi)
        spies = [0, 1] 
        civilians = [n-1, n-2] 
        
        for s in spies: values[s] = 1.0   # Kırmızı (Casus)
        for c in civilians: values[c] = 0.0 # Mavi (Sivil)
        
        return G, values, spies, civilians

    def message_passing(G, values, fixed_nodes, s_w, n_w):
        new_values = values.copy()
        for node in G.nodes():
            if node in fixed_nodes: continue 
                
            neighbors = list(G.neighbors(node))
            if not neighbors: continue
                
            neighbor_sum = sum([values[n] for n in neighbors])
            neighbor_avg = neighbor_sum / len(neighbors)
            
            # GNN Formülü: (Kendi Fikrim * w1) + (Komşu Ortalaması * w2)
            new_values[node] = (s_w * values[node]) + (n_w * neighbor_avg)
            
        return new_values

    # --- ANA AKIŞ ---
    if 'gnn_graph' not in st.session_state or st.session_state['gnn_graph'] is None:
        G, val, spies, civs = init_graph(num_nodes, connection_prob)
        st.session_state['gnn_graph'] = G
        st.session_state['initial_values'] = val # İlk hali sakla
        st.session_state['fixed_nodes'] = spies + civs

    G = st.session_state['gnn_graph']
    fixed_nodes = st.session_state['fixed_nodes']
    
    # Hesaplama (Her render'da sıfırdan hesapla ki animasyon gibi olsun)
    current_values = st.session_state['initial_values'].copy()
    for _ in range(iterations):
        current_values = message_passing(G, current_values, fixed_nodes, self_weight, neighbor_weight)

    # --- GÖRSELLEŞTİRME ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Analiz Sahası (Tur: {iterations})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.kamada_kawai_layout(G)
        
        node_colors = [current_values[n] for n in G.nodes()]
        
        # Düğümler
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.coolwarm, 
                                       node_size=600, vmin=0, vmax=1, edgecolors='black')
        # Kenarlar
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        # Etiketler (Sadece tohumları etiketle)
        labels = {n: "CASUS" if n in fixed_nodes and current_values[n]==1 else 
                     "SİVİL" if n in fixed_nodes else "" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_weight='bold')

        plt.colorbar(nodes, ax=ax, label="0 (Sivil) <----> 1 (Casus)")
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Ağ Raporu")
        spy_count = sum(1 for v in current_values.values() if v > 0.8)
        civ_count = sum(1 for v in current_values.values() if v < 0.2)
        
        st.metric("Tespit Edilen Casuslar", f"{spy_count}")
        st.metric("Güvenli Siviller", f"{civ_count}")
        
        if iterations == 0:
            st.info("Mesajlaşma başlamadı. Kaydırıcıyı artır!")
        elif iterations > 5:
            st.success("Ağ stabilize oldu. Kutuplaşma tamamlandı.")

    # --- 3. REALITY CHECK & MATH TOGGLE ---
    st.divider()
    if st.button("🔴 Kırmızı Hap: Analojiyi Kır"):
        st.session_state['math_mode_5'] = not st.session_state['math_mode_5']
        st.rerun()

    with st.expander("🛠️ Kod Müdahalesi (Reality Check)"):
        st.write("**Soru:** `Öz İrade` (self_weight) değerini **1.0** yaparsan ağda ne olur?")
        ans = st.radio("Cevap:", ["Herkes anında renk değiştirir", "Kimse fikrini değiştirmez (Donar)", "Ağ kaosa sürüklenir"])
        
        if ans == "Kimse fikrini değiştirmez (Donar)":
            st.success("Doğru! Eğer öz irade %100 ise, komşuların ne dediğinin önemi kalmaz. Bilgi yayılmaz.")
        elif ans:
            st.error("Yanlış. 1.0 demek, sadece kendi fikrini dinlemek demektir.")

if __name__ == "__main__":
    run()