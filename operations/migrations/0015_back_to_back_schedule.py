from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("operations", "0014_scraperschedule_consecutive_403_runs")]

    operations = [
        migrations.AlterField(
            model_name="scraperrun",
            name="trigger",
            field=models.CharField(
                choices=[
                    ("manual", "Manual"),
                    ("scheduled", "Scheduled"),
                    ("back_to_back", "Back to back"),
                ],
                default="manual",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="scraperschedule",
            name="last_back_to_back_finished_at",
            field=models.DateTimeField(
                blank=True,
                editable=False,
                help_text="Finish time of the most recent back-to-back scraper run.",
                null=True,
            ),
        ),
    ]
