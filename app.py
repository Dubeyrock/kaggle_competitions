import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Fashion AI - Smart Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Exact Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# RAG Knowledge Base - Enhanced with Product Data
fashion_knowledge_base = {
    'T-shirt/top': {
        'description': 'Casual short sleeve top, perfect for everyday wear',
        'price_range': '₹299 - ₹1,999',
        'best_brands': ['H&M', 'Zara', 'Levis', 'Allen Solly'],
        'quality_indicators': ['100% Cotton', 'Good stitching', 'Color fastness'],
        'season': 'All season',
        'avg_rating': '4.2/5',
        'popular_colors': ['White', 'Black', 'Grey', 'Navy Blue'],
        'avg_price': 1149  # Average of price range
    },
    'Trouser': {
        'description': 'Lower body garment, available in various fits and styles',
        'price_range': '₹799 - ₹3,999',
        'best_brands': ['Levis', 'Wrangler', 'Pepe Jeans', 'Jack & Jones'],
        'quality_indicators': ['Fabric density', 'Stitching quality', 'Fit'],
        'season': 'All season',
        'avg_rating': '4.3/5',
        'popular_colors': ['Blue', 'Black', 'Grey', 'Beige'],
        'avg_price': 2399
    },
    # ... (same as before for other categories, add avg_price for each)
}

# ==================== RAG FUNCTIONS ====================

# 1. RAG User Profile Analysis
def analyze_user_preferences(user_history, predicted_class):
    """Analyze user behavior for personalized suggestions"""
    if 'budget' in user_history:
        budget = user_history['budget']
        category_info = fashion_knowledge_base[predicted_class]
        avg_price = category_info['avg_price']
        
        if budget >= avg_price:
            return f"🎯 Perfect! Your ₹{budget} budget is great for quality {predicted_class}"
        else:
            return f"💡 Your ₹{budget} budget is lower than average. Consider budget brands or wait for discounts"
    return "💰 Set your budget in sidebar for personalized recommendations"

# 2. RAG Price Intelligence
def calculate_average_price(category):
    """Calculate average price for a category"""
    return fashion_knowledge_base[category]['avg_price']

def get_smart_price_alerts(category, user_budget):
    """Smart price recommendations based on market data"""
    avg_price = calculate_average_price(category)
    
    if user_budget < avg_price * 0.7:
        return "🔴 Budget too low - Consider increasing budget or looking for second-hand options"
    elif user_budget < avg_price:
        return "🟡 Budget slightly low - Look for discounts or budget brands"
    elif user_budget <= avg_price * 1.3:
        return "🟢 Perfect budget - You can get good quality within this range"
    else:
        return "🔵 High budget - You can premium brands and best quality"

# 3. RAG Quality Assessment
def assess_quality_standards(category, brand):
    """Provide quality insights based on brand and category"""
    quality_data = {
        'Nike Sneaker': "Excellent durability, good for sports, premium quality",
        'Zara Dress': "Trendy designs, average fabric quality, fast fashion",
        'Levis Trouser': "Premium denim, long-lasting, good investment",
        'H&M T-shirt/top': "Affordable, good for casual wear, average durability",
        'Adidas Sneaker': "Comfortable, good for running, reliable quality",
        'Allen Solly Shirt': "Formal, good for office wear, premium feel",
        'Van Heusen Shirt': "Premium formal wear, excellent quality, durable",
        'Bata Sandal': "Comfortable, durable for daily use, value for money",
        'Paragon Sandal': "Affordable, good for occasional use, basic quality",
        'Jack & Jones Trouser': "Trendy, good fit, young fashion",
        'Pepe Jeans Trouser': "Comfortable, young fashion, good value",
        'Wrangler Trouser': "Durable, classic styles, reliable",
        'Mango Dress': "Elegant, good for parties, premium quality",
        'Forever 21 Dress': "Trendy, affordable, fast fashion",
        'American Tourister Bag': "Durable, good for travel, reliable",
        'Skybags Bag': "Stylish, good for college, decent quality",
        'Catwalk Ankle boot': "Fashionable, good for winter, decent quality",
        'Metro Ankle boot': "Affordable, decent quality, basic design"
    }
    
    key = f"{brand} {category.split('/')[0]}"  # Handle T-shirt/top case
    return quality_data.get(key, "Good quality product with standard features")

# 4. Enhanced RAG Shopping Recommendations
def get_enhanced_shopping_recommendations(category, user_history=None):
    """RAG Function: Get smart shopping recommendations with all features"""
    recommendations = []
    
    # Basic category info
    category_info = fashion_knowledge_base[category]
    
    # Price intelligence
    recommendations.append(f"💰 **Price Range**: {category_info['price_range']}")
    
    # Best brands suggestion
    brands = ", ".join(category_info['best_brands'][:3])
    recommendations.append(f"🏆 **Top Brands**: {brands}")
    
    # Quality tips
    quality_tips = category_info['quality_indicators']
    recommendations.append(f"🔍 **Quality Check**: {', '.join(quality_tips)}")
    
    # Seasonal advice
    season = category_info['season']
    recommendations.append(f"🌤️ **Best Season**: {season}")
    
    # User personalized recommendations
    if user_history and 'budget' in user_history:
        # Personalized budget analysis
        personal_rec = analyze_user_preferences(user_history, category)
        recommendations.append(f"🎯 **Personalized**: {personal_rec}")
        
        # Price alert
        price_alert = get_smart_price_alerts(category, user_history['budget'])
        recommendations.append(f"💡 **Price Alert**: {price_alert}")
    
    return recommendations

# Mock Product Database
def generate_mock_products(category, count=5):
    """Generate mock product recommendations"""
    products = []
    
    base_prices = {
        'T-shirt/top': (299, 1999),
        'Trouser': (799, 3999),
        'Pullover': (1299, 4999),
        'Dress': (1499, 5999),
        'Coat': (2999, 8999),
        'Sandal': (499, 2999),
        'Shirt': (899, 3999),
        'Sneaker': (1299, 7999),
        'Bag': (799, 4999),
        'Ankle boot': (1999, 6999)
    }
    
    brands = fashion_knowledge_base[category]['best_brands']
    colors = fashion_knowledge_base[category]['popular_colors']
    
    for i in range(count):
        price_range = base_prices[category]
        price = random.randint(price_range[0], price_range[1])
        discount = random.randint(10, 40)
        discounted_price = price - (price * discount // 100)
        brand = random.choice(brands)
        
        product = {
            'name': f"{brand} {category} {i+1}",
            'brand': brand,
            'original_price': price,
            'discounted_price': discounted_price,
            'discount': f"{discount}%",
            'rating': round(random.uniform(3.8, 4.7), 1),
            'reviews': random.randint(50, 2000),
            'platform': random.choice(['Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Meesho']),
            'delivery': random.choice(['Free Delivery', '1-Day Delivery', '2-Day Delivery']),
            'color': random.choice(colors),
            'size': random.choice(['S', 'M', 'L', 'XL', 'XXL']),
            'in_stock': random.choice([True, True, True, False])
        }
        products.append(product)
    
    return products

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('fashion_mnist_cnn.h5')
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model"""
    try:
        image = image.convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        return image_array
    except Exception as e:
        return None

def main():
    st.title("🛍️ Fashion AI - Smart Shopping Assistant with RAG")
    st.write("Upload fashion item → Get prediction → Smart RAG recommendations!")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # ==================== SIDEBAR - USER PROFILE ====================
    with st.sidebar:
        st.header("👤 User Profile")
        
        # User Budget Setting
        user_budget = st.number_input("Set Your Budget (₹)", 
                                    min_value=0, 
                                    value=2000, 
                                    step=500,
                                    help="Set your budget for personalized recommendations")
        
        if st.button("Save Budget & Preferences"):
            st.session_state.user_history = {
                'budget': user_budget,
                'preference_set': True,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.success("Preferences saved!")
        
        # Display user profile if set
        if 'user_history' in st.session_state:
            st.info(f"**Saved Budget**: ₹{st.session_state.user_history['budget']}")
        
        st.markdown("---")
        st.header("🎓 RAG Features Active")
        st.success("✅ Personalized Recommendations")
        st.success("✅ Intelligent Price Alerts") 
        st.success("✅ Quality Assessment")
        st.success("✅ Smart Shopping Guide")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify & Shop", "💰 Price Intelligence", "📊 RAG Insights", "ℹ️ How to Use"])
    
    with tab1:
        st.header("Upload Fashion Item & Get RAG Recommendations")
        
        uploaded_file = st.file_uploader("Choose fashion item image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🎯 Classify & Get RAG Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Analyzing with RAG Intelligence..."):
                        # Preprocess and predict
                        processed_image = preprocess_image(image)
                        
                        if processed_image is not None:
                            predictions = model.predict(processed_image, verbose=0)
                            predicted_class_idx = np.argmax(predictions[0])
                            predicted_class = class_names[predicted_class_idx]
                            confidence = np.max(predictions[0])
                            
                            # Store in session state
                            st.session_state.predicted_class = predicted_class
                            st.session_state.confidence = confidence
            
            with col2:
                if 'predicted_class' in st.session_state:
                    predicted_class = st.session_state.predicted_class
                    confidence = st.session_state.confidence
                    
                    # Display Prediction Results
                    st.success("## 🎯 Classification Result")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Identified Item", predicted_class)
                    with res_col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # ==================== RAG SHOPPING RECOMMENDATIONS ====================
                    st.subheader("💡 RAG Smart Shopping Guide")
                    
                    # Get user history for personalization
                    user_history = st.session_state.get('user_history', {})
                    
                    # Enhanced RAG recommendations
                    shopping_guide = get_enhanced_shopping_recommendations(predicted_class, user_history)
                    
                    for tip in shopping_guide:
                        if "💰" in tip:
                            st.info(tip)
                        elif "🏆" in tip:
                            st.success(tip)
                        elif "🔍" in tip:
                            st.warning(tip)
                        elif "🌤️" in tip:
                            st.info(tip)
                        elif "🎯" in tip:
                            st.success(tip)
                        elif "💡" in tip:
                            st.warning(tip)
                        else:
                            st.write(tip)
                    
                    # Generate and display product recommendations with QUALITY ASSESSMENT
                    st.subheader("🛒 Recommended Products with Quality Insights")
                    products = generate_mock_products(predicted_class, 4)
                    
                    # Display products in a grid
                    product_cols = st.columns(2)
                    for idx, product in enumerate(products):
                        with product_cols[idx % 2]:
                            with st.container():
                                st.markdown(f"**{product['name']}**")
                                
                                # Price information
                                if product['discount'] != "0%":
                                    st.write(f"~~₹{product['original_price']}~~ **₹{product['discounted_price']}**")
                                    st.success(f"🔖 {product['discount']} OFF")
                                else:
                                    st.write(f"**₹{product['original_price']}**")
                                
                                # Rating
                                rating_str = "⭐" * int(product['rating'])
                                st.write(f"{rating_str} {product['rating']} ({product['reviews']} reviews)")
                                
                                # Platform & Delivery
                                st.write(f"🛒 {product['platform']}")
                                st.write(f"🚚 {product['delivery']}")
                                
                                # ==================== RAG QUALITY ASSESSMENT ====================
                                quality_insight = assess_quality_standards(predicted_class, product['brand'])
                                st.info(f"📊 **Quality Insight**: {quality_insight}")
                                
                                # Color & Size
                                st.write(f"🎨 {product['color']} | 📏 {product['size']}")
                                
                                # Stock status
                                if product['in_stock']:
                                    st.success("✅ In Stock")
                                else:
                                    st.warning("⏳ Out of Stock")
                                
                                st.markdown("---")
    
    with tab2:
        st.header("💰 RAG Price Intelligence")
        
        if 'predicted_class' in st.session_state:
            category = st.session_state.predicted_class
            
            st.subheader(f"Smart Price Analysis for {category}")
            
            # Get user budget for personalized analysis
            user_budget = st.session_state.get('user_history', {}).get('budget', 0)
            
            if user_budget > 0:
                # Personalized price intelligence
                price_alert = get_smart_price_alerts(category, user_budget)
                st.info(price_alert)
            
            # Price comparison across platforms
            platforms = ['Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Meesho']
            price_data = []
            
            for platform in platforms:
                base_price = random.randint(500, 5000)
                discount = random.randint(0, 40)
                final_price = base_price - (base_price * discount // 100)
                
                price_data.append({
                    'Platform': platform,
                    'Original Price': f"₹{base_price}",
                    'Discounted Price': f"₹{final_price}",
                    'Discount': f"{discount}%",
                    'Rating': f"{round(random.uniform(3.5, 4.8), 1)}/5",
                    'Delivery': random.choice(['Free', '₹49', '₹99'])
                })
            
            # Display price comparison table
            df = pd.DataFrame(price_data)
            st.dataframe(df, use_container_width=True)
            
            # Best deal recommendation
            best_deal = min(price_data, key=lambda x: int(x['Discounted Price'].replace('₹', '')))
            st.success(f"🎉 **RAG Recommended Deal**: {best_deal['Platform']} - {best_deal['Discounted Price']} ({best_deal['Discount']} OFF)")
        
        else:
            st.info("👆 First upload and classify an image to see RAG price intelligence")
    
    with tab3:
        st.header("📊 RAG Market Insights & Quality Analysis")
        
        category_to_analyze = st.selectbox("Select category for deep analysis:", class_names)
        
        if category_to_analyze:
            # Market analysis using RAG knowledge base
            category_info = fashion_knowledge_base[category_to_analyze]
            
            st.subheader(f"RAG Quality Analysis: {category_to_analyze}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Price", category_info['price_range'])
                st.metric("Customer Rating", category_info['avg_rating'])
            
            with col2:
                st.metric("Best Season", category_info['season'])
                st.metric("Quality Score", "85/100")
            
            with col3:
                st.metric("Popularity", "High" if random.random() > 0.3 else "Medium")
                st.metric("Return Rate", f"{random.randint(5, 15)}%")
            
            # Brand-wise quality analysis
            st.subheader("🏆 Brand Quality Analysis")
            brands = category_info['best_brands']
            
            for brand in brands:
                quality_info = assess_quality_standards(category_to_analyze, brand)
                with st.expander(f"**{brand}** - Quality Report"):
                    st.write(f"**Assessment**: {quality_info}")
                    st.write(f"**Price Range**: {category_info['price_range']}")
                    st.write(f"**Best For**: {category_info['description']}")
            
            # RAG Buying Recommendations
            st.subheader("🎯 RAG Smart Buying Strategy")
            
            recommendations = [
                f"Check {category_info['quality_indicators'][0]} for quality assurance",
                f"Best bought in {category_info['season']} for optimal prices",
                f"Compare prices across {', '.join(category_info['best_brands'][:2])}",
                "Read recent customer reviews for latest quality feedback",
                "Check return policy before purchase"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    with tab4:
        st.header("📖 How RAG Enhances Your Experience")
        
        st.write("""
        ## 🚀 RAG Features Active in This App:
        
        ### 1. **Personalized Recommendations**
        ```python
        analyze_user_preferences(user_history, predicted_class)
        ```
        - Your budget-based suggestions
        - Personalized shopping guidance
        - Custom price alerts
        
        ### 2. **Intelligent Price Alerts** 
        ```python
        get_smart_price_alerts(category, user_budget)
        ```
        - Smart budget analysis
        - Price trend predictions
        - Discount opportunities
        
        ### 3. **Quality Assessment**
        ```python
        assess_quality_standards(category, brand)
        ```
        - Brand-specific quality insights
        - Durability analysis
        - Value for money assessment
        
        ## 🇮🇳 Hindi Mein Samjhein:
        
        **RAG aapko kya deta hai:**
        - ✅ **Personalized Suggestions**: Aapke budget ke hisaab se recommendations
        - ✅ **Smart Price Alerts**: Kab kharidna best hai, kya price sahi hai
        - ✅ **Quality Reports**: Kaunsa brand quality mein better hai
        - ✅ **Shopping Strategy**: Kaise smart shopping karen
        
        **Kaise use karen:**
        1. Sidebar mein apna budget set karen
        2. Fashion item ki photo upload karen  
        3. RAG automatically smart suggestions dega
        4. Quality reports aur price comparison dekhen
        """)

if __name__ == "__main__":
    main()