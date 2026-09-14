# เอกสารประกอบ Pipeline: ตั้งค่า (Config) และ โหลดข้อมูล

เอกสารนี้อธิบายรายละเอียดของ 2 ส่วนใน pipeline ทำนายสถานะปั๊มน้ำ (Tanzania Water Pump)
คือส่วน **ตั้งค่า (Config)** และ **โหลดข้อมูล (Load Data)**

---

## 1. ตั้งค่า (Config)

ส่วนนี้รวมค่าที่ปรับได้ทั้งหมดไว้ในที่เดียว เพื่อให้ทีมแก้ไขพฤติกรรมของ pipeline
ได้โดยไม่ต้องไปไล่แก้โค้ดหลายจุด

### Path ของไฟล์

```python
DATA_DIR = Path("data")

TRAIN_VALUES_PATH = DATA_DIR / "train_values.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
TEST_VALUES_PATH = DATA_DIR / "test_values.csv"

ID_COL = "id"
TARGET_COL = "status_group"
DATE_COL = "date_recorded"
```

กำหนดตำแหน่งไฟล์ input ทั้งหมด รวมถึงชื่อคอลัมน์สำคัญ (id, target, วันที่)
ที่ถูกอ้างอิงซ้ำหลายจุดในโค้ด เพื่อไม่ให้เกิดปัญหาพิมพ์ชื่อคอลัมน์ผิดเวลาแก้ไข

### คอลัมน์ที่ตัดทิ้งก่อนสร้างโมเดล — `DROP_COLS`

| คอลัมน์ | เหตุผลที่ตัดทิ้ง |
|---|---|
| `recorded_by` | มีค่าเดียวทั้งคอลัมน์ ไม่มี signal อะไรเลย |
| `wpt_name`, `subvillage`, `ward`, `scheme_name` | cardinality สูงมาก (คล้ายข้อความอิสระ) เก็บไว้เสี่ยง overfitting โดยไม่ได้ signal เพิ่มมากในเวอร์ชัน baseline |
| `num_private` | เป็น 0 มากกว่า 98% ของข้อมูล แทบไม่มีความหลากหลาย |
| `quantity_group`, `extraction_type`, `extraction_type_group`, `waterpoint_type_group`, `payment_type`, `source_type` | ซ้ำซ้อนกับคอลัมน์พี่น้องที่เก็บไว้แล้ว (เช่น `quantity_group` ซ้ำกับ `quantity`) |

### คอลัมน์ตัวเลข — `NUMERIC_COLS`

```python
NUMERIC_COLS = [
    "amount_tsh", "gps_height", "longitude", "latitude",
    "population", "construction_year", "region_code", "district_code",
]
```

คอลัมน์เหล่านี้จะถูกเติมค่าขาดหาย (missing) ด้วยค่ามัธยฐานตอนสร้าง preprocessing pipeline ทีหลัง

### คอลัมน์ categorical — `CATEGORICAL_COLS`

```python
CATEGORICAL_COLS = [
    "funder", "installer", "basin", "region", "lga",
    "public_meeting", "scheme_management", "permit",
    "extraction_type_class", "management", "management_group",
    "payment", "water_quality", "quality_group", "quantity",
    "source", "source_class", "waterpoint_type",
]
```

คอลัมน์เหล่านี้จะถูกเข้ารหัสแบบ one-hot encoding ทีหลัง

### การจัดการ cardinality สูง — `HIGH_CARDINALITY_COLS`

```python
HIGH_CARDINALITY_COLS = ["funder", "installer", "lga"]
MIN_CATEGORY_FREQUENCY = 20
```

คอลัมน์ `funder`, `installer`, `lga` มีค่าที่ไม่ซ้ำกันหลักพันค่า ถ้า one-hot encode ตรงๆ
จะทำให้จำนวนฟีเจอร์บวมมาก จึงกำหนดว่า **ค่าที่พบน้อยกว่า 20 ครั้งในข้อมูล train**
จะถูกรวมเป็นหมวด `"other"` (ควรเรียนรู้ความถี่จากข้อมูล train เท่านั้น เพื่อไม่ให้เกิด data leakage)

### ตั้งค่าทั่วไป

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
```

- `RANDOM_STATE` — ตรึงค่าเพื่อให้ผลลัพธ์ reproduce ได้ทุกครั้งที่รัน
- `TEST_SIZE = 0.2` — สำรองไว้ใช้ตอนแบ่งข้อมูล train 20% เป็น validation set สำหรับวัดผลโมเดล

### ฟีเจอร์ที่จะสร้างเพิ่มเอง

```python
ENGINEERED_NUMERIC_COLS = ["recorded_year", "recorded_month", "pump_age"]
```

รายชื่อฟีเจอร์ที่วางแผนจะสร้างขึ้นใหม่ในขั้นตอน feature engineering (ยังไม่มีอยู่ในไฟล์ CSV ต้นฉบับ)
ประกาศไว้ล่วงหน้าเป็นจุดเริ่มต้น เพราะภายหลังต้องนำไปรวมกับ `NUMERIC_COLS` ตอนกำหนดฟีเจอร์ทั้งหมดให้โมเดล

---

## 2. โหลดข้อมูล (Load Data)

### โหลดข้อมูล train

```python
def load_train_data():
    """โหลดและรวม train_values.csv กับ train_labels.csv โดยใช้ id"""
    values = pd.read_csv(TRAIN_VALUES_PATH)
    labels = pd.read_csv(TRAIN_LABELS_PATH)
    df = values.merge(labels, on=ID_COL, how="inner")
    return df
```

ข้อมูล train ของ dataset นี้แยกเป็น 2 ไฟล์: `train_values.csv` (ฟีเจอร์ต่างๆ ของปั๊มน้ำ)
และ `train_labels.csv` (คำตอบ `status_group`) ฟังก์ชันนี้จะรวม 2 ไฟล์เข้าด้วยกันโดยใช้ `id`
เป็นตัวเชื่อม (`inner join` เพื่อให้แน่ใจว่าทุกแถวที่ได้มีทั้งฟีเจอร์และคำตอบครบ)

### โหลดข้อมูล test

```python
def load_test_data():
    """โหลด test_values.csv (ไม่มี label)"""
    return pd.read_csv(TEST_VALUES_PATH)
```

ข้อมูล test มีแค่ฟีเจอร์ ไม่มีคำตอบ (เพราะเป็นชุดที่ต้องทำนาย) จึงโหลดตรงๆ
ไม่ต้อง merge กับอะไร

### วิธีใช้

```python
df_train_raw = load_train_data()
print(f"โหลดข้อมูล train ได้ {len(df_train_raw):,} แถว")
df_train_raw.head()
```

หลังโหลดแล้วควรเช็คการกระจายตัวของ label ก่อนเสมอ เพื่อดูว่าคลาสไหนไม่สมดุล:

```python
df_train_raw["status_group"].value_counts()
```

จะเห็นว่า `functional` และ `non functional` มีสัดส่วนมาก ส่วน `functional needs repair`
มีสัดส่วนน้อย (~7%) — เป็นข้อมูลสำคัญที่ควรพิจารณาตอนเลือกโมเดล/ปรับ `class_weight` ในขั้นตอนถัดไป

---

## ขั้นตอนถัดไป (ทีมทำต่อจากนี้)

- **Feature engineering**: ใช้ `ENGINEERED_NUMERIC_COLS` เป็นจุดเริ่มสร้างฟีเจอร์ใหม่ เช่น
  แปลง `date_recorded` เป็นปี/เดือน, คำนวณอายุปั๊มจาก `construction_year`
- **จัดการ missing/sentinel values**: หลายคอลัมน์ใช้ `0` แทนค่า "ไม่ทราบ" เช่น `gps_height`,
  `population`, `construction_year` — ควรแปลงเป็น `NaN` ก่อนส่งเข้าโมเดล
- **สร้าง Pipeline preprocessing + model**: ใช้ `NUMERIC_COLS`, `CATEGORICAL_COLS`,
  `HIGH_CARDINALITY_COLS` ที่ประกาศไว้ด้านบนเป็นตัวกำหนด column mapping
- **เทรน/ประเมินผล/ทำนาย test set**
