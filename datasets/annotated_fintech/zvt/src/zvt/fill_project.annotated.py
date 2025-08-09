# script to auto generate some files
# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies
from zvt.autocode.generator import gen_kdata_schema, gen_exports

# ✅ Best Practice: Group imports from the same module on a single line for readability
from zvt.contract import AdjustType
from zvt.contract import IntervalLevel

# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns and dependencies


def gen_kdata_schemas():
    """
    generate kdata(OHLC) schemas for tradable entity

    """
    # A股行情
    gen_kdata_schema(
        pkg="zvt",
        providers=["em", "qmt", "joinquant"],
        entity_type="stock",
        levels=[
            # 🧠 ML Signal: Repeated pattern of calling gen_kdata_schema with different parameters
            level
            for level in IntervalLevel
            if level not in (IntervalLevel.LEVEL_L2_QUOTE, IntervalLevel.LEVEL_TICK)
        ],
        adjust_types=[None, AdjustType.hfq],
        entity_in_submodule=True,
    )
    # 中国期货
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="future",
        levels=[IntervalLevel.LEVEL_1DAY],
        entity_in_submodule=True,
    )

    # 美股
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="stockus",
        levels=[IntervalLevel.LEVEL_1DAY],
        adjust_types=[None, AdjustType.hfq],
        entity_in_submodule=True,
    )
    # 美指
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="indexus",
        levels=[IntervalLevel.LEVEL_1DAY],
        entity_in_submodule=True,
    )

    # 港股
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="stockhk",
        levels=[IntervalLevel.LEVEL_1DAY],
        adjust_types=[None, AdjustType.hfq],
        entity_in_submodule=True,
    )

    # 板块行情
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="block",
        levels=[
            IntervalLevel.LEVEL_1DAY,
            IntervalLevel.LEVEL_1WEEK,
            IntervalLevel.LEVEL_1MON,
        ],
        entity_in_submodule=True,
    )

    # ⚠️ SAST Risk (Low): Direct execution of code in the main block
    # 🧠 ML Signal: Exporting factors from a package
    # A股指数行情
    gen_kdata_schema(
        pkg="zvt",
        providers=["em", "sina"],
        entity_type="index",
        levels=[IntervalLevel.LEVEL_1DAY, IntervalLevel.LEVEL_1WEEK],
        entity_in_submodule=True,
    )

    # etf行情
    gen_kdata_schema(
        pkg="zvt",
        providers=["sina"],
        entity_type="etf",
        levels=[IntervalLevel.LEVEL_1DAY],
        entity_in_submodule=True,
    )

    # currency行情
    gen_kdata_schema(
        pkg="zvt",
        providers=["em"],
        entity_type="currency",
        levels=[IntervalLevel.LEVEL_1DAY],
        entity_in_submodule=True,
    )


if __name__ == "__main__":
    # gen_exports("api")
    # gen_exports("broker")
    # gen_exports("common")
    # gen_exports("contract", export_from_package=True, export_modules=["schema"])
    # gen_exports("domain", export_from_package=True)
    gen_exports("factors", export_from_package=True)
    # gen_exports("trading")

    # gen_exports("ml")
    # gen_exports("utils", export_from_package=False, export_var=True)
    # gen_exports('informer')
    # gen_exports('trader')
    # gen_exports('autocode')
    # gen_exports("zhdate")
    # gen_exports("recorders", export_from_package=True, exclude_modules=["qmt"])
    # gen_exports("tag", export_from_package=False)
    # gen_kdata_schemas()
    # zip_dir(ZVT_TEST_DATA_PATH, zip_file_name=DATA_SAMPLE_ZIP_PATH)
