import pandas as pd
from sklearn.datasets import load_iris

class DataLoader:
    
    def __init__(self):
        # هنا نجهز المتغيرات الأساسية ونحمل الداتا الجاهزة
        self.data = None
        
        # تحميل dataset الخاص بالـ Iris من sklearn
        self.iris = load_iris()

    def load_data(self):
        # نحول البيانات إلى جدول (DataFrame) عشان يصير التعامل معها أسهل
        self.data = pd.DataFrame(
            self.iris.data,
            columns=self.iris.feature_names
        )
        
        # نضيف العمود الخاص بالتصنيفات (0,1,2)
        self.data['target'] = self.iris.target
        
        # نحول الأرقام إلى أسماء الأنواع الحقيقية للزهور
        self.data['species'] = [
            self.iris.target_names[i] for i in self.iris.target
        ]

        # بس تأكيد بسيط إن البيانات تحملت بدون مشاكل
        print("✅ تم تحميل البيانات بنجاح!")
        
        # نرجع الداتا عشان تستخدمها باقي الكلاسات
        return self.data

    def get_summary(self):
        # نتأكد أول إن البيانات محملة قبل ما نطلع الإحصائيات
        if self.data is None:
            print("⚠️  لم يتم تحميل البيانات بنجاح")
            return None
        
        # نعرض ملخص سريع عن البيانات (متوسط، أقل، أعلى قيم...)
        return self.data.describe()
