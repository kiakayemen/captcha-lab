from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("operations", "0015_back_to_back_schedule")]

    operations = [
        migrations.AddField(
            model_name="scraperschedule",
            name="back_to_back_enabled",
            field=models.BooleanField(
                default=True,
                help_text=(
                    "Run the scraper back to back from 10:00 to 11:00 Tehran time. "
                    "When disabled, the interval schedule runs normally all day."
                ),
            ),
        ),
    ]
