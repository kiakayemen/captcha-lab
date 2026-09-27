from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("operations", "0013_proxyendpointhealth")]

    operations = [
        migrations.AddField(
            model_name="scraperschedule",
            name="consecutive_403_runs",
            field=models.PositiveIntegerField(
                default=0,
                editable=False,
                help_text="Consecutive scheduled runs ending in HTTP 403.",
            ),
        ),
    ]
