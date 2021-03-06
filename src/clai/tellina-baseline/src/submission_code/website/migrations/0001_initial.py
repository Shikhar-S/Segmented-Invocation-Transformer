# Generated by Django 3.0.5 on 2020-04-06 01:39

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Annotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('submission_time', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
        migrations.CreateModel(
            name='AnnotationUpdate',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('update_str', models.TextField()),
                ('update_type', models.TextField(default='nl')),
                ('submission_time', models.DateTimeField(default=django.utils.timezone.now)),
                ('status', models.TextField(default='open')),
                ('annotation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.Annotation')),
            ],
        ),
        migrations.CreateModel(
            name='Command',
            fields=[
                ('str', models.TextField(primary_key=True, serialize=False)),
                ('template', models.TextField(default='')),
                ('language', models.TextField(default='bash')),
            ],
        ),
        migrations.CreateModel(
            name='Comment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('str', models.TextField()),
                ('submission_time', models.DateTimeField(default=django.utils.timezone.now)),
                ('reply', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='website.Comment')),
            ],
        ),
        migrations.CreateModel(
            name='NL',
            fields=[
                ('str', models.TextField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('str', models.TextField(primary_key=True, serialize=False)),
                ('annotations', models.ManyToManyField(to='website.Annotation')),
                ('commands', models.ManyToManyField(to='website.Command')),
            ],
        ),
        migrations.CreateModel(
            name='Translation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.FloatField()),
                ('num_upvotes', models.PositiveIntegerField(default=0)),
                ('num_downvotes', models.PositiveIntegerField(default=0)),
                ('num_stars', models.PositiveIntegerField(default=0)),
                ('nl', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.NL')),
                ('pred_cmd', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.Command')),
            ],
        ),
        migrations.CreateModel(
            name='URL',
            fields=[
                ('str', models.TextField(primary_key=True, serialize=False)),
                ('html_content', models.TextField(default='')),
                ('commands', models.ManyToManyField(to='website.Command')),
                ('tags', models.ManyToManyField(to='website.Tag')),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('access_code', models.TextField(default='')),
                ('ip_address', models.TextField(default='')),
                ('first_name', models.TextField(default='anonymous')),
                ('last_name', models.TextField(default='anonymous')),
                ('organization', models.TextField(default='--')),
                ('city', models.TextField(default='--')),
                ('region', models.TextField(default='--')),
                ('country', models.TextField(default='--')),
                ('is_annotator', models.BooleanField(default=False)),
                ('is_judger', models.BooleanField(default=False)),
                ('time_logged', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Vote',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ip_address', models.TextField(default='')),
                ('upvoted', models.BooleanField(default=False)),
                ('downvoted', models.BooleanField(default=False)),
                ('starred', models.BooleanField(default=False)),
                ('translation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.Translation')),
            ],
        ),
        migrations.CreateModel(
            name='URLTag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.TextField()),
                ('url', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.URL')),
            ],
        ),
        migrations.CreateModel(
            name='Notification',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.TextField(default='comment')),
                ('creation_time', models.DateTimeField(default=django.utils.timezone.now)),
                ('status', models.TextField(default='issued')),
                ('annotation_update', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='website.AnnotationUpdate')),
                ('comment', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='website.Comment')),
                ('receiver', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='notification_receiver', to='website.User')),
                ('sender', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='notification_sender', to='website.User')),
                ('url', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.URL')),
            ],
        ),
        migrations.CreateModel(
            name='NLRequest',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('submission_time', models.DateTimeField(default=django.utils.timezone.now)),
                ('nl', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.NL')),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='website.User')),
            ],
        ),
        migrations.AddField(
            model_name='comment',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.User'),
        ),
        migrations.AddField(
            model_name='command',
            name='tags',
            field=models.ManyToManyField(to='website.Tag'),
        ),
        migrations.AddField(
            model_name='annotationupdate',
            name='comment',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.Comment'),
        ),
        migrations.AddField(
            model_name='annotationupdate',
            name='judger',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.User'),
        ),
        migrations.CreateModel(
            name='AnnotationProgress',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.TextField()),
                ('annotator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.User')),
                ('tag', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='website.Tag')),
                ('url', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.URL')),
            ],
        ),
        migrations.AddField(
            model_name='annotation',
            name='annotator',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.User'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='cmd',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.Command'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='nl',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.NL'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='url',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='website.URL'),
        ),
    ]
