from django.contrib import admin
from edgeapp.models import Address
# Register your models here.

@admin.register(Address)
class CodeAdmin(admin.ModelAdmin):
    pass