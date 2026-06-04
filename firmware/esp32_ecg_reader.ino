#define ECG_PIN 34

void setup()
{
    Serial.begin(115200);
}

void loop()
{
    int ecg = analogRead(ECG_PIN);

    Serial.println(ecg);

    delay(10);
}