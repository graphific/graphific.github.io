---
---

hey

```ruby
{% codeblock lang:ruby title:"Check if a number is prime" mark:3 %}
class Fixnum
  def prime?
    ('1' * self) !~ /^1?$|^(11+?)\1+$/
  end
end
{% endcodeblock %}
```

guys?
