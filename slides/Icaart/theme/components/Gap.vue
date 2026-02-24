<template>
    <div class="border-2 border-black rounded-lg bg-gray-300 flex flex-row">
        <div class="flex items-center justify-center p-2">
            <img src="/figures/icons/gap.drawio.svg" class="w-30 " />
        </div>
        <div class="p-2 ">
            <p class="break-full">
                <slot mdc-unwrap="p" />
            </p>
        </div>
    </div>
</template>