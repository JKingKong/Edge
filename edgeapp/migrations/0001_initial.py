# Generated by Django 2.1.1 on 2019-06-07 01:36

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Address',
            fields=[
                ('aid', models.AutoField(primary_key=True, serialize=False)),
                ('address', models.CharField(max_length=200, unique=True)),
                ('threshold', models.IntegerField(default=5)),
            ],
        ),
    ]