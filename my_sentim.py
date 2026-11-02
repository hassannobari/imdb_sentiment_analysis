import streamlit as st
import joblib

#################load
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_AI.pkl')
    vectorizer = joblib.load('tfid_vectorizer.pkl')
    return model , vectorizer

model , vectorizer = load_model()
#####################ui

st.title("جمله تو خوبه یا بد! بیا بفهمیم🐚")
st.write("هر جمله ای رو که میخوایی به انگلیسی وارد کن تا بهت بگم مثبته یا منفی 🫴")
st.write("امتحانش مجانیه 🤫")
st.write("..این سایت از نظرات سایت ای ام دی بی برای تحلیل استفاده میکند لذا ترجیحا هنگام وارد کردن متن از متن های بلند استفاده کنید")

user_text = st.text_area("حالا متن خودتو اینجا بنویس.....",height=300)
if st.button("بریم پیش بینی کنیم......"):
    if user_text.strip()== "":
        st.warning("اول متن بنویس بعدش این دکمه رو بزن...")
    else:
        text_vectorized = vectorizer.transform([user_text])
        prediction = model.predict(text_vectorized)

        probability = model.predict_proba(text_vectorized)[0]
        confidence = probability[prediction].item()*100

        if prediction == 1:
            st.success("متنی که تو نوشتی به احتمال زیاد مثبته👍")
            st.success(f"درصد اطمینان: {confidence:.2f}%")
            st.video('ok.mp4',autoplay=True,loop=True,muted=True)
            st.toast("woooooooow!!")
        else:
            st.error("متنی که نوشتی احتمال زیاد منفیه!!!")
            st.error(f"درصد اطمینان :  {confidence:.2f}")
            st.video('sad.mp4',autoplay=True,loop=True,muted=True)
            st.toast("oooooh!!")
