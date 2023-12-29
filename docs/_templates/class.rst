{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. class:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
    :nosignatures:
    :toctree: _method
    :template: method.rst
   {% for item in methods %}
      {% if item != "__init__" %}
      {%- if item not in inherited_members %}
         ~{{ name }}.{{ item }}
      {% endif %}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
