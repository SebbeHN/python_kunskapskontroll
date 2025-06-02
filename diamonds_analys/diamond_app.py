import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="💎 Guldfynds Diamond Data Analysis",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)


import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Ladda och förbehandla diamantdata"""

    # Hämta korrekt sökväg oavsett om appen körs lokalt eller i molnet
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'diamonds.csv')

    try:
        df = pd.read_csv(csv_path).copy()

        # Skapa scoring system
        cut_scores = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        color_scores = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
        clarity_scores = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

        df['cut_score'] = df['cut'].map(cut_scores)
        df['color_score'] = df['color'].map(color_scores)
        df['clarity_score'] = df['clarity'].map(clarity_scores)

        df['quality_score'] = (
            df['cut_score'] * 0.4 +
            df['color_score'] * 0.3 +
            df['clarity_score'] * 0.3
        ) / 7 * 5

        df['price_per_carat'] = df['price'] / df['carat']
        df['value_score'] = df['quality_score'] / (df['price_per_carat'] / 1000)

        def price_segment(price):
            if price < 1000: return 'Budget (< $1K)'
            elif price < 2500: return 'Standard ($1K-$2.5K)'
            elif price < 5000: return 'Premium ($2.5K-$5K)'
            elif price < 10000: return 'Luxury ($5K-$10K)'
            else: return 'Ultra-Luxury (> $10K)'

        df['price_segment'] = df['price'].apply(price_segment)

        return df

    except FileNotFoundError:
        st.error(f"❌ diamonds.csv inte hittad i {csv_path}")
        return None



df = load_data()

if df is not None:
    
    
    st.title("💎 Guldfynd Market Data Analysis")
    st.markdown("### *Comprehensive Analysis of Diamond Dataset*")
    
    
    st.info("📋 **Context:** Analysis of diamond market data to understand pricing patterns, quality distributions, and value opportunities for potential market entry.")
    
    
    st.sidebar.title("🔧 Analysis Controls")
    
    
    page = st.sidebar.selectbox(
        "📍 Select Analysis:",
        ["🏠 Dataset Overview", "🔍 Interactive Explorer", "💰 Value Analysis", "🏆 Key Insights"]
    )
    
   
    st.sidebar.subheader("🎛️ Data Filters")
    
    
    budget_range = st.sidebar.slider(
        "💰 Price Range ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(500, 10000),
        step=100
    )
    
    
    carat_range = st.sidebar.slider(
        "⚖️ Carat Range",
        min_value=float(df['carat'].min()),
        max_value=float(df['carat'].max()),
        value=(0.3, 2.0),
        step=0.1
    )
    
    
    filtered_df = df[
        (df['price'] >= budget_range[0]) & 
        (df['price'] <= budget_range[1]) &
        (df['carat'] >= carat_range[0]) & 
        (df['carat'] <= carat_range[1])
    ]
    
    
    
    if page == "🏠 Dataset Overview":
        
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💎 Total Diamonds",
                f"{len(filtered_df):,}",
                f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
            )
        
        with col2:
            avg_price = filtered_df.loc[:, 'price'].mean()
            st.metric(
                "💰 Average Price",
                f"${avg_price:,.0f}",
                f"{(avg_price - df.loc[:, 'price'].mean())/df.loc[:, 'price'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        with col3:
            avg_carat = filtered_df.loc[:, 'carat'].mean()
            st.metric(
                "⚖️ Average Carat",
                f"{avg_carat:.2f}",
                f"{(avg_carat - df.loc[:, 'carat'].mean())/df.loc[:, 'carat'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        with col4:
            avg_quality = filtered_df.loc[:, 'quality_score'].mean()
            st.metric(
                "⭐ Quality Score",
                f"{avg_quality:.2f}/5",
                f"{(avg_quality - df.loc[:, 'quality_score'].mean())/df.loc[:, 'quality_score'].mean()*100:+.1f}%" if len(filtered_df) != len(df) else None
            )
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Distribution")
            fig_price = px.histogram(
                filtered_df, 
                x='price', 
                nbins=50,
                title="Distribution of Diamond Prices",
                color_discrete_sequence=['#1f77b4']
            )
            fig_price.update_layout(
                xaxis_title="Price (USD)",
                yaxis_title="Number of Diamonds",
                showlegend=False
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("💎 Market Segments")
            segment_counts = filtered_df['price_segment'].value_counts()
            fig_segment = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Market Share by Price Segment"
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        
        
        st.subheader("📈 Price Correlation Analysis")
        
        
        corr_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'quality_score', 'price']
        corr_matrix = filtered_df[corr_cols].corr()
        price_corr = corr_matrix['price'].drop('price').abs().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Strongest price correlations:**")
            for factor, corr in price_corr.head(5).items():
                st.write(f"• **{factor}**: {corr:.3f}")
            
            st.write("**Key Insights:**")
            st.write("• Carat is the strongest price driver")
            st.write("• Physical dimensions closely follow carat")
            st.write("• Quality score shows moderate correlation")
            st.write("• Depth/table have minimal price impact")
        
        with col2:
            
            top_factor = price_corr.index[0]
            if top_factor in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df.sample(min(5000, len(filtered_df))),
                    x=top_factor,
                    y='price',
                    color='quality_score',
                    title=f"Price vs {top_factor} (colored by quality)",
                    opacity=0.6
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        
        st.subheader("💎 The Four Cs Analysis")
        
        four_c_col1, four_c_col2 = st.columns(2)
        
        with four_c_col1:
            
            cut_analysis = filtered_df.groupby('cut', observed=False).agg({
                'price': ['mean', 'count'],
                'carat': 'mean'
            }).round(2)
            cut_analysis.columns = ['Avg_Price', 'Count', 'Avg_Carat']
            
            st.markdown("**Cut Quality Distribution:**")
            for cut in cut_analysis.index:
                data = cut_analysis.loc[cut]
                st.write(f"• **{cut}**: ${data['Avg_Price']:,.0f} avg price, {data['Count']} diamonds")
        
        with four_c_col2:
            
            color_analysis = filtered_df.groupby('color', observed=False).agg({
                'price': 'mean',
                'price_per_carat': 'mean'
            }).round(0)
            
            st.markdown("**Color Grade Analysis:**")
            for color in sorted(color_analysis.index):
                data = color_analysis.loc[color]
                st.write(f"• **{color}**: ${data['price']:,.0f} avg price, ${data['price_per_carat']:,.0f}/carat")
    
    elif page == "🔍 Interactive Explorer":
        
        st.title("🔍 Interactive Data Explorer")
        st.markdown("Explore relationships between different diamond characteristics")
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "📊 X-axis:",
                ['carat', 'price', 'depth', 'table', 'quality_score', 'value_score']
            )
        
        with col2:
            y_axis = st.selectbox(
                "📊 Y-axis:",
                ['price', 'carat', 'quality_score', 'value_score', 'price_per_carat']
            )
        
        with col3:
            color_by = st.selectbox(
                "🎨 Color by:",
                ['cut', 'color', 'clarity', 'price_segment', 'quality_score']
            )
        
        
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
        
        
        st.subheader("⭐ Quality Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            cut_counts = filtered_df['cut'].value_counts()
            fig_cut = px.bar(
                x=cut_counts.index,
                y=cut_counts.values,
                title="Distribution of Cut Quality",
                color=cut_counts.values,
                color_continuous_scale='viridis'
            )
            fig_cut.update_layout(showlegend=False, xaxis_title="Cut", yaxis_title="Count")
            st.plotly_chart(fig_cut, use_container_width=True)
        
        with col2:
            
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
    
    elif page == "💰 Value Analysis":
        
        st.title("💰 Value-for-Money Analysis")
        st.markdown("### *Find diamonds with the best value proposition*")
        
        
        with st.expander("ℹ️ How Value Score Works"):
            st.markdown("""
            **Value Score Calculation:**
            - Quality Score = Weighted average of Cut (40%), Color (30%), Clarity (30%)
            - Value Score = Quality Score / (Price per Carat / 1000)
            - Higher values indicate better quality relative to price
            """)
        
        
        st.subheader("🏆 Best Value Diamonds")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            best_values = filtered_df.nlargest(15, 'value_score')[
                ['carat', 'cut', 'color', 'clarity', 'price', 'value_score', 'quality_score']
            ].round(2)
            
            
            display_df = best_values.copy()
            display_df['Price'] = display_df['price'].apply(lambda x: f"${x:,.0f}")
            display_df['Value Score'] = display_df['value_score'].apply(lambda x: f"{x:.2f}")
            display_df['Quality Score'] = display_df['quality_score'].apply(lambda x: f"{x:.2f}/5")
            
            final_display = display_df[['carat', 'cut', 'color', 'clarity', 'Price', 'Value Score', 'Quality Score']]
            st.dataframe(final_display, use_container_width=True)
        
        with col2:
            
            st.subheader("📊 Value Statistics")
            
            avg_value = filtered_df['value_score'].mean()
            top_10_threshold = filtered_df['value_score'].quantile(0.9)
            high_value_count = len(filtered_df[filtered_df['value_score'] > top_10_threshold])
            
            st.metric("Average Value Score", f"{avg_value:.2f}")
            st.metric("Top 10% Threshold", f"{top_10_threshold:.2f}")
            st.metric("High Value Diamonds", f"{high_value_count:,}")
            
            
            st.markdown("**💡 Value Insights:**")
            best_value_cut = best_values['cut'].mode().iloc[0] if len(best_values) > 0 else 'N/A'
            best_value_color = best_values['color'].mode().iloc[0] if len(best_values) > 0 else 'N/A'
            
            st.write(f"• Most common cut in top values: **{best_value_cut}**")
            st.write(f"• Most common color in top values: **{best_value_color}**")
            st.write(f"• Sweet spot size: **{best_values['carat'].mean():.2f} carats**")
        
        
        st.subheader("📈 Value Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_value_dist = px.histogram(
                filtered_df,
                x='value_score',
                nbins=50,
                title="Distribution of Value Scores"
            )
            fig_value_dist.add_vline(
                x=filtered_df['value_score'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Average"
            )
            st.plotly_chart(fig_value_dist, use_container_width=True)
        
        with col2:
            
            fig_value_price = px.scatter(
                filtered_df.sample(min(3000, len(filtered_df))),
                x='value_score',
                y='price',
                color='quality_score',
                size='carat',
                title="Value Score vs Price",
                opacity=0.6
            )
            st.plotly_chart(fig_value_price, use_container_width=True)
    
    elif page == "🏆 Key Insights":
        
        st.title("🏆 Key Insights & Recommendations")
        st.markdown("### *Data-driven conclusions from the analysis*")
        
        
        st.subheader("📋 Executive Summary")
        
        
        total_diamonds = len(df)
        avg_price = df['price'].mean()
        price_carat_corr = df['price'].corr(df['carat'])
        most_common_cut = df['cut'].mode().iloc[0]
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            **📊 Dataset Overview:**
            - **{total_diamonds:,}** diamonds analyzed
            - Average price: **${avg_price:,.0f}**
            - Price range: **${df['price'].min():,}** - **${df['price'].max():,}**
            - Carat range: **{df['carat'].min():.2f}** - **{df['carat'].max():.2f}**
            """)
        
        with summary_col2:
            st.markdown(f"""
            **🔍 Key Findings:**
            - Carat drives **{price_carat_corr:.1%}** of price variation
            - Most common cut: **{most_common_cut}**
            - Premium segment represents largest opportunity
            - Quality scoring reveals undervalued diamonds
            """)
        
        
        st.subheader("📈 Detailed Analysis")
        
        insight_tabs = st.tabs(["💰 Pricing", "💎 Quality", "📊 Market", "🎯 Opportunities"])
        
        with insight_tabs[0]:  
            st.markdown("""
            **💰 Pricing Insights:**
            
            **Primary Price Drivers:**
            1. **Carat Weight** - Strongest correlation (0.92+)
            2. **Physical Dimensions** - Directly related to carat
            3. **Quality Factors** - Cut, Color, Clarity (moderate impact)
            
            **Price Patterns:**
            - Exponential relationship between carat and price
            - Premium for "magic sizes" (0.5, 1.0, 1.5, 2.0 carats)
            - Quality improvements show diminishing returns
            
            **Market Efficiency:**
            - Most diamonds priced fairly relative to characteristics
            - Opportunities exist in undervalued high-quality stones
            - Value-conscious buyers should focus on cut optimization
            """)
        
        with insight_tabs[1]:  
            premium_cut_pct = (df['cut'] == 'Premium').mean() * 100
            ideal_cut_pct = (df['cut'] == 'Ideal').mean() * 100
            
            st.markdown(f"""
            **💎 Quality Distribution Insights:**
            
            **Cut Quality:**
            - **{ideal_cut_pct:.1f}%** are Ideal cut (highest quality)
            - **{premium_cut_pct:.1f}%** are Premium cut
            - Very Good cut offers best value proposition
            
            **Color Grading:**
            - G-H colors provide optimal value/quality balance
            - D-F colors command significant premium
            - Color preference varies by setting and personal taste
            
            **Clarity Levels:**
            - SI1-VS2 range offers best value for most buyers
            - Higher clarities often imperceptible to naked eye
            - Investment grade stones typically VVS1+
            """)
        
        with insight_tabs[2]:  
            budget_pct = (df['price'] < 1000).mean() * 100
            premium_pct = ((df['price'] >= 2500) & (df['price'] < 5000)).mean() * 100
            
            st.markdown(f"""
            **📊 Market Segmentation:**
            
            **Price Segments:**
            - **Budget (< $1K)**: {budget_pct:.1f}% of market
            - **Premium ($2.5K-$5K)**: {premium_pct:.1f}% of market
            - **Luxury ($5K+)**: Smaller volume, higher margins
            
            **Market Characteristics:**
            - Mainstream market focuses on value optimization
            - Premium segment balances size and quality
            - Luxury segment prioritizes perfection
            
            **Consumer Behavior:**
            - Size preference drives many purchasing decisions
            - Quality education increases value awareness
            - Brand and certification important for trust
            """)
        
        with insight_tabs[3]:  
            if 'value_score' in df.columns:
                high_value_opportunities = len(df[df['value_score'] > df['value_score'].quantile(0.9)])
                
            st.markdown(f"""
            **🎯 Strategic Opportunities:**
            
            **For Market Entry:**
            - Focus on Premium segment ($2.5K-$5K) for volume
            - Target 0.7-1.0 carat range for optimal demand
            - Emphasize Very Good+ cut quality
            - Offer G-H color grades for value positioning
            
            **For Value Optimization:**
            - Identify diamonds just below "magic sizes"
            - Focus on excellent cut over perfect color/clarity
            - Consider fancy shapes for differentiation
            - Leverage certification for quality assurance
            
            **For Investment:**
            - Larger stones (2+ carats) show stronger appreciation
            - Ideal cut + D-F color + VVS+ clarity for long-term value
            - Rare characteristics command premiums
            - Provenance and documentation increasingly important
            """)
        
        
        st.subheader("🎯 Final Recommendations")
        
        st.success("""
        **Key Takeaways:**
        1. **Carat weight is king** - Focus on size optimization within budget
        2. **Cut quality matters most** - Don't compromise on cut for other factors
        3. **Smart compromises** - G-H color and SI1-VS2 clarity offer best value
        4. **Market timing** - Avoid "magic sizes" for better pricing
        5. **Quality scoring** - Use data-driven approaches to identify value
        """)
    
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        💎 Diamond Data Analysis | Comprehensive Market Insights 📊
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Could not load data. Make sure diamonds.csv is in the same directory as the app.")
    st.info("To run the app: `streamlit run diamond_app.py`")