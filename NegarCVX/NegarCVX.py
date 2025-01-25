import cvxpy as cp
import numpy as np
import math
from sympy import symbols, sympify, Le
import sympy as sp
import random



def define_uncertain_problem(opt_var,E,B,d,constraints_str,obj_func_type,obj_func):
    """
    مشخصات مساله عدم قطعیت را مشخص می کند و سپس آن را حل کرده و جواب را بر می گرداند.

    Parameters:
    opt_var (string): 'x' متغیر بهینه سازی را بصورت متنی مثل
    E (float): متغیر ریسک
    B (float): متغیر اطمینان
    d (int): اندازه بردار ضرایب تابع هزینه
    constraints_str (list): محدودیتهای مساله به همراه متغیرهای عدم قطعیت
    obj_func_type (string): نوع بهینه سازی مینیم سازی یا ماکسیمم سازی
    obj_func (string): تابع بهینه سازی
    Returns:
            محدودیتهای جدید که دیگر دارای مقادیر مشخص هستند را بر می گرداند.


    Example:
    >>> opt_var='x'
    >>> E=0.1
    >>> B=0.6
    >>> d=15
    >>> constraints_str = [
        "x - v + u <= 2.2",
        "x + 3 + u <= 2"
        ]
    >>> obj_func_type='min'
    >>> obj_func= ''
    >>> uncertanity_constraints(constraints_str)
    """
    assert isinstance(opt_var, (str)) and len(opt_var) > 0 , "مقدار متغیر بهینه سازی به درستی مشخص نشده است."
    assert isinstance(E, (int, float)) and E > 0 and E<=1, "مقدار باید عددی احتمالی باشد."
    assert isinstance(B, (int, float)) and B > 0 and B<=1, "مقدار باید عددی احتمالی باشد."
    assert isinstance(d, (int)) and d > 0 , "مقدار باید عددی طبیعی و بزرگتر از صفر باشد."
    assert isinstance(constraints_str, (list)) and len(constraints_str) > 0 , "محدودیتها باید مشخص شده باشد."
    assert isinstance(obj_func_type, (str)) and len(obj_func_type) > 0 , "نوع بهینه سازی به درستی مشخص نشده است."
    assert isinstance(obj_func, (str)) and len(obj_func) > 0 , "تابع بهینه سازی به درستی مشخص نشده است."

    symbols_sympy = sp.symbols(opt_var)  # می‌توانید هر تعداد سمبل تعریف کنید

    # اگر symbols_sympy فقط یک سمبل باشد، آن را به یک لیست تبدیل می‌کنیم
    if not isinstance(symbols_sympy, (list, tuple)):
        symbols_sympy = [symbols_sympy]

    # بررسی محدودیت‌ها و انتخاب آن‌هایی که شامل حداقل یکی از symbols_sympy هستند
    uncert_constraints = []
    for expr in constraints_str:
        # تبدیل رشته محدودیت به یک عبارت sympy
        expr_sympy = sp.sympify(expr)
        
        # بررسی آیا عبارت شامل حداقل یکی از symbols_sympy است
        if any(symbol in expr_sympy.free_symbols for symbol in symbols_sympy):
            uncert_constraints.append(expr_sympy)

    print(uncert_constraints)
    # استخراج خودکار تمام متغیرهای سمبلیک از محدودیت‌ها
    all_symbols = set()
    for expr in uncert_constraints:
        all_symbols.update(expr.free_symbols)

    print(all_symbols)

    # تعداد تکرارهای مورد نیاز
    N=math.ceil((2/E)*(math.log(B,math.e)+d-1))

    print('Number of scenarios is:',N)

    print('Number of constraint will be:', N*len(uncert_constraints))

    # لیست برای ذخیره محدودیت‌های جدید
    constraints_sympy = []

    # حلقه برای تولید محدودیت‌های جدید
    for i in range(N):
        for item in uncert_constraints:
            new_item = replace_uncertain_vars_with_random_digit(all_symbols,item,opt_var)
            constraints_sympy.append(new_item)

    
    assert isinstance(constraints_sympy, (list)) and len(constraints_sympy) > 0 , "محدودیتها خالی است."
    
    print('constraints after :',constraints_sympy)

    # تعریف متغیرهای cvxpy به صورت پویا
    variables_cvxpy = [cp.Variable() for _ in symbols_sympy]

    # ایجاد نگاشت خودکار از سمبل‌های sympy به متغیرهای cvxpy
    sympy_to_cvxpy = {sym: var for sym, var in zip(symbols_sympy, variables_cvxpy)}

    # تبدیل خودکار تمام محدودیت‌ها
    constraints_cvxpy = []
    for c in constraints_sympy:
        try:
            constraints_cvxpy.append(convert_sympy_to_cvxpy(c, sympy_to_cvxpy))
        except ValueError as e:
            print(f"خطا در تبدیل محدودیت: {e}")

    # چاپ محدودیت‌های تبدیل‌شده برای دیباگ
    #print("محدودیت‌های cvxpy:")
    #for c in constraints_cvxpy:
    #    print(c)

    #print('variables_cvxpy=',variables_cvxpy)
    
    


    # تجزیه و تحلیل رشته obj_func به یک عبارت سمبلیک
    expr = sp.sympify(obj_func)
    
    # استخراج تمام متغیرهای سمبلیک از عبارت
    symbols = expr.free_symbols
    
    # ایجاد یک دیکشنری برای نگاشت متغیرهای سمبلیک به متغیرهای cvxpy
    cvxpy_vars = {symbol: cp.Variable() for symbol in symbols}
    
    # تبدیل عبارت sympy به عبارت cvxpy
    expr_cvxpy = convert_sympy_to_cvxpy2(expr, cvxpy_vars)
    
    # تعیین نوع تابع بهینه‌سازی
    if obj_func_type == 'min':
        objective = cp.Minimize(expr_cvxpy)
    else:
        objective = cp.Maximize(expr_cvxpy)
    
    # تعریف مسئله بهینه‌سازی
    problem = cp.Problem(objective,constraints_cvxpy)
    
    # حل مسئله
    problem.solve()
    
    # بازگرداندن نتایج
    results = {str(var): value.value for var, value in cvxpy_vars.items()}
    results['optimal_value'] = problem.value
    results['status'] = problem.status
    
    return results
    

# تابع برای جایگزینی متغیرهای نامشخص با اعداد تصادفی
def replace_uncertain_vars_with_random_digit(all_symbols,constraint,opt_var):
    # تولید اعداد تصادفی برای هر متغیر
    random_values = {symbol: random.uniform(0, 5) for symbol in all_symbols if symbol.name not in opt_var}
    # جایگزینی متغیرها با اعداد تصادفی
    new_constraint = constraint.subs(random_values)
    return new_constraint
    

# تابع برای تبدیل خودکار محدودیت‌های sympy به cvxpy
def convert_sympy_to_cvxpy(constraint, sympy_to_cvxpy):
    # بررسی اینکه آیا محدودیت یک عبارت رابطه‌ای است
    if not isinstance(constraint, sp.core.relational.Relational):
        raise ValueError(f"محدودیت {constraint} یک عبارت رابطه‌ای معتبر نیست.")

    left = constraint.lhs  # سمت چپ محدودیت
    right = constraint.rhs # سمت راست محدودیت
    op = constraint.rel_op # عملگر مقایسه

    # تابع بازگشتی برای تبدیل عبارات sympy به cvxpy
    def convert_expression(expr):
        if expr in sympy_to_cvxpy:
            return sympy_to_cvxpy[expr]  # تبدیل سمبل‌های sympy به متغیرهای cvxpy
        elif expr.is_Number:
            return float(expr)  # تبدیل اعداد sympy به float
        elif expr.is_Add:  # اگر عبارت جمع باشد
            return sum(convert_expression(arg) for arg in expr.args)
        elif expr.is_Mul:  # اگر عبارت ضرب باشد
            return cp.multiply(convert_expression(expr.args[0]), convert_expression(expr.args[1]))
        elif expr.is_Pow:  # اگر عبارت توان باشد
            return cp.power(convert_expression(expr.args[0]), convert_expression(expr.args[1]))
        else:
            raise ValueError(f"عبارت {expr} پشتیبانی نمی‌شود.")

    # تبدیل سمت چپ و راست محدودیت
    left_cp = convert_expression(left)
    right_cp = convert_expression(right)

    # ایجاد محدودیت cvxpy
    if op == '<=':
        return left_cp <= right_cp
    elif op == '>=':
        return left_cp >= right_cp
    elif op == '==':
        return left_cp == right_cp
    else:
        raise ValueError(f"عملگر {op} پشتیبانی نمی‌شود.")

# تابع بازگشتی برای تبدیل عبارات sympy به cvxpy
def convert_sympy_to_cvxpy2(expr, cvxpy_vars):
    """
    تبدیل یک عبارت sympy به یک عبارت cvxpy.
    """
    if expr.is_constant():
        return float(expr)
    elif expr.is_symbol:
        return cvxpy_vars[expr]
    elif expr.is_Add:
        return sum(convert_sympy_to_cvxpy2(arg, cvxpy_vars) for arg in expr.args)
    elif expr.is_Mul:
        return sp.prod([convert_sympy_to_cvxpy2(arg, cvxpy_vars) for arg in expr.args])
    elif expr.is_Pow:
        base, exponent = expr.args
        if exponent == 2:  # اگر توان ۲ باشد، از cp.square استفاده می‌کنیم
            return cp.square(convert_sympy_to_cvxpy2(base, cvxpy_vars))
        else:
            raise ValueError("توان‌های غیر از ۲ پشتیبانی نمی‌شوند.")
    else:
        raise ValueError(f"عبارت {expr} پشتیبانی نمی‌شود.")
    