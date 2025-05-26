import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurera sidan
st.set_page_config(
    page_title="💎 Diamond Market Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Ladda och förbehandla diamantdata"""
    try:
        # Ladda data och skapa en explicit kopia
        df = pd.read_csv('diamonds.csv').copy()
        
        # Skapa scoring system
        cut_scores = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        color_scores = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
        clarity_scores = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
        
        # Använd .loc för säker tilldelning
        df.loc[:, 'cut_score'] = df['cut'].map(cut_scores)
        df.loc[:, 'color_score'] = df['color'].map(color_scores)
        df.loc[:, 'clarity_score'] = df['clarity'].map(clarity_scores)
        
        # Kvalitetspoäng (viktad)
        df.loc[:, 'quality_score'] = (
            df['cut_score'] * 0.4 + 
            df['color_score'] * 0.3 + 
            df['clarity_score'] * 0.3
        ) / 7 * 5
        
        # Beräkna pris per karat
        df.loc[:, 'price_per_carat'] = df['price'] / df['carat']
        
        # Value score
        df.loc[:, 'value_score'] = df['quality_score'] / (df['price_per_carat'] / 1000)
        
        # Prissegment
        def price_segment(price):
            if price < 1000: return 'Budget (< $1K)'
            elif price < 2500: return 'Standard ($1K-$2.5K)'
            elif price < 5000: return 'Premium ($2.5K-$5K)'
            elif price < 10000: return 'Luxury ($5K-$10K)'
            else: return 'Ultra-Luxury (> $10K)'
        
        df.loc[:, 'price_segment'] = df['price'].apply(price_segment)
        
        return df
    except FileNotFoundError:
        st.error("❌ diamonds.csv inte hittad! Lägg filen i samma mapp som appen.")
        return None

# Ladda data
df = load_data()

if df is not None:
    
    # SIDEBAR - Filters och Navigation
    st.sidebar.title("🔧 Kontroller")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📍 Välj sida:",
        ["🏠 Översikt", "🔍 Interaktiv Explorer", "💰 Prisguide", "🏆 Rekommendationer"]
    )
    
    # Gemensamma filter
    st.sidebar.subheader("🎛️ Filter")
    
    # Budget filter
    budget_range = st.sidebar.slider(
        "💰 Budget ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(500, 10000),
        step=100
    )
    
    # Karat filter
    carat_range = st.sidebar.slider(
        "⚖️ Karat",
        min_value=float(df['carat'].min()),
        max_value=float(df['carat'].max()),
        value=(0.3, 2.0),
        step=0.1
    )
    
    # Filtrera data
    filtered_df = df[
        (df['price'] >= budget_range[0]) & 
        (df['price'] <= budget_range[1]) &
        (df['carat'] >= carat_range[0]) & 
        (df['carat'] <= carat_range[1])
    ]
    
    # MAIN CONTENT baserat på vald sida
    
    if page == "🏠 Översikt":
        
        st.title("💎 Diamond Market Analyzer")
        st.markdown("### *Datadriven analys av diamantmarknaden*")
        
        # Key metrics i kolumner
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💎 Totalt antal",
                f"{len(filtered_df):,}",
                f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
            )
        
        with col2:
            avg_price = filtered_df['price'].mean()
            st.metric(
                "💰 Medelpris",
                f"${avg_price:,.0f}",
                f"{(avg_price - df['price'].mean())/df['price'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        with col3:
            avg_carat = filtered_df['carat'].mean()
            st.metric(
                "⚖️ Medelkarat",
                f"{avg_carat:.2f}",
                f"{(avg_carat - df['carat'].mean())/df['carat'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        with col4:
            avg_quality = filtered_df['quality_score'].mean()
            st.metric(
                "⭐ Kvalitetspoäng",
                f"{avg_quality:.2f}/5",
                f"{(avg_quality - df['quality_score'].mean())/df['quality_score'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        # Huvudvisualiseringar
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prisfördelning")
            fig_price = px.histogram(
                filtered_df, 
                x='price', 
                nbins=50,
                title="Fördelning av diamantpriser",
                color_discrete_sequence=['#1f77b4']
            )
            fig_price.update_layout(
                xaxis_title="Pris (USD)",
                yaxis_title="Antal diamanter",
                showlegend=False
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("💎 Prissegment")
            segment_counts = filtered_df['price_segment'].value_counts()
            fig_segment = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Marknadsandel per segment"
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        
        # Korrelationsanalys
        st.subheader("📈 Prisdriven Faktorer")
        
        # Beräkna korrelationer
        corr_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'quality_score', 'price']
        corr_matrix = filtered_df[corr_cols].corr()
        price_corr = corr_matrix['price'].drop('price').abs().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Starkaste korrelationer med pris:**")
            for factor, corr in price_corr.head(5).items():
                st.write(f"• **{factor}**: {corr:.3f}")
        
        with col2:
            # Scatter plot av starkaste korrelationen
            top_factor = price_corr.index[0]
            if top_factor in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df.sample(min(5000, len(filtered_df))),  # Sample för prestanda
                    x=top_factor,
                    y='price',
                    color='quality_score',
                    title=f"Pris vs {top_factor} (färg = kvalitet)",
                    opacity=0.6
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif page == "🔍 Interaktiv Explorer":
        
        st.title("🔍 Interaktiv Diamond Explorer")
        
        # Kontroller för explorern
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "📊 X-axel:",
                ['carat', 'price', 'depth', 'table', 'quality_score', 'value_score']
            )
        
        with col2:
            y_axis = st.selectbox(
                "📊 Y-axel:",
                ['price', 'carat', 'quality_score', 'value_score', 'price_per_carat']
            )
        
        with col3:
            color_by = st.selectbox(
                "🎨 Färgkoda:",
                ['cut', 'color', 'clarity', 'price_segment', 'quality_score']
            )
        
        # Skapa scatter plot
        sample_size = min(5000, len(filtered_df))
        plot_df = filtered_df.sample(sample_size)
        
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            size='carat' if x_axis != 'carat' and y_axis != 'carat' else None,
            hover_data=['cut', 'color', 'clarity', 'price', 'carat'],
            title=f"{y_axis.title()} vs {x_axis.title()}"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Kvalitetsfördelning
        st.subheader("⭐ Kvalitetsanalys")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cut fördelning
            cut_counts = filtered_df['cut'].value_counts()
            fig_cut = px.bar(
                x=cut_counts.index,
                y=cut_counts.values,
                title="Fördelning av Cut-kvalitet",
                color=cut_counts.values,
                color_continuous_scale='viridis'
            )
            fig_cut.update_layout(showlegend=False, xaxis_title="Cut", yaxis_title="Antal")
            st.plotly_chart(fig_cut, use_container_width=True)
        
        with col2:
            # Color vs Clarity heatmap
            heatmap_data = pd.crosstab(filtered_df['color'], filtered_df['clarity'])
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                title="Color vs Clarity Distribution",
                color_continuous_scale='Blues',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    elif page == "💰 Prisguide":
        
        st.title("💰 Smart Prisguide")
        st.markdown("### *Hitta bästa värdet för din budget*")
        
        # Budget selector
        budget_guide = st.slider(
            "🎯 Din budget ($):",
            min_value=500,
            max_value=20000,
            value=5000,
            step=500
        )
        
        # Filtrera på budget
        budget_diamonds = df[df['price'] <= budget_guide]
        
        if len(budget_diamonds) > 0:
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bästa värden för budgeten
                st.subheader(f"🏆 Bästa värden för ${budget_guide:,}")
                
                best_values = budget_diamonds.nlargest(10, 'value_score')[
                    ['carat', 'cut', 'color', 'clarity', 'price', 'value_score', 'quality_score']
                ].round(2)
                
                # Gör tabellen mer visuell
                best_values['💰 Pris'] = best_values['price'].apply(lambda x: f"${x:,.0f}")
                best_values['⭐ Värde'] = best_values['value_score'].apply(lambda x: f"{x:.2f}")
                best_values['🏆 Kvalitet'] = best_values['quality_score'].apply(lambda x: f"{x:.2f}/5")
                
                display_df = best_values[['carat', 'cut', 'color', 'clarity', '💰 Pris', '⭐ Värde', '🏆 Kvalitet']]
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                # Budget stats
                st.subheader("📊 Budget Statistics")
                
                avg_carat_budget = budget_diamonds['carat'].mean()
                avg_quality_budget = budget_diamonds['quality_score'].mean()
                count_budget = len(budget_diamonds)
                best_value_score = budget_diamonds['value_score'].max()
                
                st.metric("💎 Tillgängliga", f"{count_budget:,}")
                st.metric("⚖️ Snitt karat", f"{avg_carat_budget:.2f}")
                st.metric("⭐ Snitt kvalitet", f"{avg_quality_budget:.2f}/5")
                st.metric("🏆 Bästa värde", f"{best_value_score:.2f}")
            
            # Priskomparison per karat-intervall
            st.subheader("📈 Pris per Karat-intervall")
            
            # Skapa karat-bins
            budget_diamonds_copy = budget_diamonds.copy()
            budget_diamonds_copy.loc[:, 'carat_bin'] = pd.cut(
                budget_diamonds_copy['carat'], 
                bins=[0, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0],
                labels=['<0.5', '0.5-0.7', '0.7-1.0', '1.0-1.5', '1.5-2.0', '>2.0']
            )
            
            price_by_carat = budget_diamonds_copy.groupby('carat_bin', observed=False).agg({
                'price': ['mean', 'count'],
                'value_score': 'mean'
            }).round(2)
            
            price_by_carat.columns = ['Medelpris', 'Antal', 'Värde_Score']
            price_by_carat = price_by_carat.reset_index()
            
            fig_price_carat = px.bar(
                price_by_carat,
                x='carat_bin',
                y='Medelpris',
                color='Värde_Score',
                title="Medelpris per Karat-intervall",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_price_carat, use_container_width=True)
            
            # Smarta tips
            st.subheader("💡 Smarta Köptips")
            
            tips_col1, tips_col2 = st.columns(2)
            
            with tips_col1:
                st.markdown("""
                **🎯 Optimera din budget:**
                - Sök diamanter strax under "magiska" storlekar (0.9ct vs 1.0ct)
                - Very Good cut ger 90% av Ideal's prestanda
                - G-H färg är sweet spot för värde
                - SI1 klarhet är ofta perfekt för ögat
                """)
            
            with tips_col2:
                # Visa budget-specifika rekommendationer
                if budget_guide < 2000:
                    rec = "Budget-tips: Fokusera på cut och undvik extremt små diamanter"
                elif budget_guide < 5000:
                    rec = "Standard-tips: Balansera mellan storlek och kvalitet"
                elif budget_guide < 10000:
                    rec = "Premium-tips: Nu kan du få både storlek OCH kvalitet"
                else:
                    rec = "Lyx-tips: Sikta på Ideal cut, D-F färg, VVS+ klarhet"
                
                st.info(f"**För din budget:** {rec}")
        
        else:
            st.warning("Inga diamanter hittades för denna budget. Prova att öka budgeten.")
    
    elif page == "🏆 Rekommendationer":
        
        st.title("🏆 Personliga Rekommendationer")
        
        # Användarpreferenser
        st.subheader("🎯 Dina Preferenser")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_budget = st.number_input(
                "💰 Max budget ($):",
                min_value=500,
                max_value=50000,
                value=3000,
                step=500
            )
        
        with col2:
            priority = st.selectbox(
                "🎯 Prioritet:",
                ["Bästa värde", "Största storlek", "Högsta kvalitet", "Balanserat"]
            )
        
        with col3:
            min_carat = st.number_input(
                "⚖️ Min karat:",
                min_value=0.2,
                max_value=3.0,
                value=0.5,
                step=0.1
            )
        
        # Filtrera baserat på preferenser
        user_diamonds = df[
            (df['price'] <= user_budget) & 
            (df['carat'] >= min_carat)
        ]
        
        if len(user_diamonds) > 0:
            
            # Sortera baserat på prioritet
            if priority == "Bästa värde":
                recommendations = user_diamonds.nlargest(15, 'value_score')
                sort_metric = "Value Score"
            elif priority == "Största storlek":
                recommendations = user_diamonds.nlargest(15, 'carat')
                sort_metric = "Karat"
            elif priority == "Högsta kvalitet":
                recommendations = user_diamonds.nlargest(15, 'quality_score')
                sort_metric = "Kvalitetspoäng"
            else:  # Balanserat
                user_diamonds_copy = user_diamonds.copy()
                user_diamonds_copy.loc[:, 'balanced_score'] = (
                    user_diamonds_copy['value_score'] * 0.4 +
                    user_diamonds_copy['quality_score'] * 0.3 +
                    (user_diamonds_copy['carat'] / user_diamonds_copy['carat'].max()) * 5 * 0.3
                )
                recommendations = user_diamonds_copy.nlargest(15, 'balanced_score')
                sort_metric = "Balanserad Score"
            
            # Visa rekommendationer
            st.subheader(f"🎯 Dina Top 15 Rekommendationer (sorterat på {sort_metric})")
            
            # Formatera för visning
            display_recs = recommendations[[
                'carat', 'cut', 'color', 'clarity', 'price', 
                'value_score', 'quality_score'
            ]].copy()
            
            display_recs['Price'] = display_recs['price'].apply(lambda x: f"${x:,}")
            display_recs['Value'] = display_recs['value_score'].apply(lambda x: f"{x:.2f}")
            display_recs['Quality'] = display_recs['quality_score'].apply(lambda x: f"{x:.2f}/5")
            
            final_display = display_recs[['carat', 'cut', 'color', 'clarity', 'Price', 'Value', 'Quality']]
            
            # Highlighta top 3
            styled_df = final_display.head(15).style.apply(
                lambda x: ['background-color: #90EE90' if i < 3 else '' for i in range(len(x))],
                axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Personlig analys
            st.subheader("📊 Personlig Marknadsanalys")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Tillgängliga alternativ
                st.metric("💎 Tillgängliga alternativ", f"{len(user_diamonds):,}")
                st.metric("⚖️ Genomsnittlig karat", f"{user_diamonds['carat'].mean():.2f}")
                st.metric("💰 Genomsnittligt pris", f"${user_diamonds['price'].mean():,.0f}")
                
                # Bästa fynd
                best_deal = recommendations.iloc[0]
                st.success(f"""
                **🏆 Ditt bästa fynd:**
                {best_deal['carat']:.2f}ct {best_deal['cut']} {best_deal['color']} {best_deal['clarity']}
                Pris: ${best_deal['price']:,} | Värde: {best_deal['value_score']:.2f}
                """)
            
            with col2:
                # Visuell fördelning av dina alternativ
                fig_user = px.scatter(
                    user_diamonds.sample(min(1000, len(user_diamonds))),
                    x='carat',
                    y='price',
                    color='value_score',
                    title="Dina alternativ: Karat vs Pris",
                    color_continuous_scale='RdYlGn',
                    opacity=0.7
                )
                
                # Highlighta top rekommendationer
                fig_user.add_scatter(
                    x=recommendations.head(5)['carat'],
                    y=recommendations.head(5)['price'],
                    mode='markers',
                    marker=dict(size=15, color='red', line=dict(width=2, color='black')),
                    name='Top 5 Rekommendationer'
                )
                
                st.plotly_chart(fig_user, use_container_width=True)
        
        else:
            st.warning("Inga diamanter matchar dina kriterier. Prova att justera budget eller storlek.")
    
    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        💎 Diamond Market Analyzer | Byggd med Streamlit och kärlek för data 📊
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Kunde inte ladda data. Se till att diamonds.csv finns i samma mapp som appen.")
    st.info("För att köra appen: `streamlit run app.py`")